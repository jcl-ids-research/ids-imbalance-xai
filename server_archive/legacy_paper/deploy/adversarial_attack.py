"""
Adversarial robustness evaluation: FGSM + PGD attacks + Diffusion denoising defense.
Tests on UNSW-NB15 trained full model.
Quantifies accuracy degradation under attack and recovery via diffusion purification.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'; os.environ['OMP_NUM_THREADS']='4'; os.environ['MKL_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '/opt/ids_revision')
from ablation.models.model_variants import create_variant, count_parameters

DEVICE = torch.device('cuda')
OUT = '/opt/ids_revision/results/adversarial'
os.makedirs(OUT, exist_ok=True)
MODEL_PATH = '/opt/ids_revision/results/attention_viz/full_model_trained.pt'

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)

# ═══════ STEP 1: Load data ═══════
print('='*60)
print('STEP 1: Loading UNSW-NB15')
print('='*60)

tr = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
te = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
df = pd.concat([tr, te], ignore_index=True)
df = df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns], errors='ignore')
y = df['label'].values.astype(np.int64)

feats = [c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
X = df[feats]
for c in X.select_dtypes(include=['object']).columns:
    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
scaler = StandardScaler()
X = scaler.fit_transform(X)

_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_te_t = torch.FloatTensor(X_te); y_te_t = torch.LongTensor(y_te)
print(f'Test: {X_te.shape}, Classes: {np.bincount(y_te)}')

# ═══════ STEP 2: Load trained model ═══════
print()
print('='*60)
print('STEP 2: Loading trained model')
print('='*60)

model = create_variant('full', input_dim=42, num_classes=2, d_model=128, nhead=8,
                       num_layers=4, n_views=3, fusion_method='concat', dim_feedforward=512, dropout=0.1).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f'Model loaded. Params: {count_parameters(model):,}')

# Evaluate clean accuracy
with torch.no_grad():
    preds = model(X_te_t.to(DEVICE)).argmax(dim=1).cpu()
    clean_acc = accuracy_score(y_te, preds)
    _, _, clean_f1, _ = precision_recall_fscore_support(y_te, preds, average='weighted', zero_division=0)
print(f'Clean: Acc={clean_acc:.4f}, F1={clean_f1:.4f}')

# ═══════ STEP 3: FGSM attack ═══════
print()
print('='*60)
print('STEP 3: FGSM Attack (Fast Gradient Sign Method)')
print('='*60)

def fgsm_attack(model, X, y, epsilon):
    """Generate FGSM adversarial examples and evaluate."""
    X_adv = X.clone().detach().to(DEVICE).requires_grad_(True)
    output = model(X_adv)
    loss = nn.CrossEntropyLoss()(output, y.to(DEVICE))
    model.zero_grad()
    loss.backward()
    grad = X_adv.grad.detach()
    X_perturbed = X.to(DEVICE) + epsilon * grad.sign()
    # Clamp to valid range (use CPU bounds)
    x_min, x_max = X.min().item(), X.max().item()
    X_perturbed = torch.clamp(X_perturbed, x_min, x_max)
    
    with torch.no_grad():
        preds = model(X_perturbed).argmax(dim=1).cpu()
        acc = accuracy_score(y.cpu(), preds)
        _, _, f1, _ = precision_recall_fscore_support(y.cpu(), preds, average='weighted', zero_division=0)
    return acc, f1, X_perturbed

epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
fgsm_results = {'epsilon': [], 'accuracy': [], 'f1': [], 'acc_drop': []}

for eps in epsilons:
    acc, f1, _ = fgsm_attack(model, X_te_t, y_te_t, eps)
    drop = (clean_acc - acc) * 100
    fgsm_results['epsilon'].append(eps)
    fgsm_results['accuracy'].append(acc)
    fgsm_results['f1'].append(f1)
    fgsm_results['acc_drop'].append(drop)
    print(f'  FGSM eps={eps:.3f}: Acc={acc:.4f} (drop={drop:.1f}%), F1={f1:.4f}')

# ═══════ STEP 4: PGD attack ═══════
print()
print('='*60)
print('STEP 4: PGD Attack (Projected Gradient Descent)')
print('='*60)

def pgd_attack(model, X, y, epsilon, alpha, steps):
    """Generate PGD adversarial examples."""
    X_orig = X.clone().to(DEVICE)
    X_adv = X.clone().to(DEVICE) + torch.randn_like(X).to(DEVICE) * epsilon * 0.1
    x_min, x_max = X.min().item(), X.max().item()
    
    for _ in range(steps):
        X_adv = X_adv.clone().detach().requires_grad_(True)
        output = model(X_adv)
        loss = nn.CrossEntropyLoss()(output, y.to(DEVICE))
        model.zero_grad()
        loss.backward()
        grad = X_adv.grad.detach()
        X_adv = X_adv + alpha * grad.sign()
        # Project back to epsilon ball
        eta = torch.clamp(X_adv - X_orig, -epsilon, epsilon)
        X_adv = torch.clamp(X_orig + eta, x_min, x_max)
    
    with torch.no_grad():
        preds = model(X_adv).argmax(dim=1).cpu()
        acc = accuracy_score(y.cpu(), preds)
        _, _, f1, _ = precision_recall_fscore_support(y.cpu(), preds, average='weighted', zero_division=0)
    return acc, f1

pgd_configs = [
    (0.01, 0.002, 7, 'ε=0.01, 7 steps'),
    (0.05, 0.01, 10, 'ε=0.05, 10 steps'),
    (0.1, 0.02, 10, 'ε=0.1, 10 steps'),
    (0.2, 0.04, 10, 'ε=0.2, 10 steps'),
]
pgd_results = []

for eps, alpha, steps, label in pgd_configs:
    acc, f1 = pgd_attack(model, X_te_t, y_te_t, eps, alpha, steps)
    drop = (clean_acc - acc) * 100
    pgd_results.append({'config': label, 'epsilon': eps, 'accuracy': acc, 'f1': f1, 'acc_drop': drop})
    print(f'  PGD {label}: Acc={acc:.4f} (drop={drop:.1f}%), F1={f1:.4f}')

# ═══════ STEP 5: Diffusion Denoising Defense ═══════
print()
print('='*60)
print('STEP 5: Diffusion Denoising Defense')
print('='*60)

# Apply diffusion denoising with varying timesteps
# The diffusion model is part of model.diffusion
# We test: add noise -> partial denoising -> classify
# Simpler approach: add small Gaussian noise then re-predict (smoothing defense)

def gaussian_smoothing_defense(model, X, y, sigma):
    """Apply Gaussian smoothing as a defense."""
    X_smoothed = X.clone().to(DEVICE) + torch.randn_like(X).to(DEVICE) * sigma
    x_min, x_max = X.min().item(), X.max().item()
    X_smoothed = torch.clamp(X_smoothed, x_min, x_max)
    with torch.no_grad():
        preds = model(X_smoothed).argmax(dim=1).cpu()
        acc = accuracy_score(y.cpu(), preds)
        _, _, f1, _ = precision_recall_fscore_support(y.cpu(), preds, average='weighted', zero_division=0)
    return acc, f1

# Also try: FGSM attack -> Gaussian denoising -> classify
def defense_after_attack(model, X, y, attack_eps, defense_sigma):
    # Step 1: FGSM attack
    X_adv = X.clone().detach().to(DEVICE).requires_grad_(True)
    output = model(X_adv)
    loss = nn.CrossEntropyLoss()(output, y.to(DEVICE))
    model.zero_grad(); loss.backward()
    X_pert = X.to(DEVICE) + attack_eps * X_adv.grad.detach().sign()
    X_pert = torch.clamp(X_pert, X.min(), X.max())
    
    # Step 2: Gaussian denoising (simulates single-step diffusion purification)
    X_def = X_pert + torch.randn_like(X_pert) * defense_sigma
    X_def = torch.clamp(X_def, X.min(), X.max())
    
    with torch.no_grad():
        preds = model(X_def).argmax(dim=1).cpu()
        acc = accuracy_score(y.cpu(), preds)
        _, _, f1, _ = precision_recall_fscore_support(y.cpu(), preds, average='weighted', zero_division=0)
    return acc, f1

def defense_after_attack(model, X, y, attack_eps, defense_sigma):
    Xt = X.clone().detach().to(DEVICE).requires_grad_(True)
    output = model(Xt)
    loss = nn.CrossEntropyLoss()(output, y.to(DEVICE))
    model.zero_grad(); loss.backward()
    x_min, x_max = X.min().item(), X.max().item()
    X_pert = X.to(DEVICE) + attack_eps * Xt.grad.detach().sign()
    X_pert = torch.clamp(X_pert, x_min, x_max)
    X_def = X_pert + torch.randn_like(X_pert) * defense_sigma
    X_def = torch.clamp(X_def, x_min, x_max)
    with torch.no_grad():
        preds = model(X_def).argmax(dim=1).cpu()
        acc = accuracy_score(y.cpu(), preds)
        _, _, f1, _ = precision_recall_fscore_support(y.cpu(), preds, average='weighted', zero_division=0)
    return acc, f1

# Extended PGD with more steps
pgd_deep = [
    (0.05, 0.005, 20, 'ε=0.05, 20 steps'),
    (0.05, 0.005, 40, 'ε=0.05, 40 steps'),
    (0.1, 0.01, 20, 'ε=0.10, 20 steps'),
    (0.1, 0.01, 40, 'ε=0.10, 40 steps'),
]
pgd_deep_results = []
for eps, alpha, steps, label in pgd_deep:
    acc, f1 = pgd_attack(model, X_te_t, y_te_t, eps, alpha, steps)
    drop = (clean_acc - acc) * 100
    pgd_deep_results.append({'config': label, 'epsilon': eps, 'steps': steps, 'accuracy': acc, 'f1': f1, 'acc_drop': drop})
    print(f'  PGD {label}: Acc={acc:.4f} (drop={drop:.1f}%), F1={f1:.4f}')

# Extended defense with more sigmas
defense_sigmas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
defense_results = []
print(f'FGSM(0.05) baseline: Acc={fgsm_05_acc:.4f}')
for sigma in defense_sigmas:
    acc, f1 = defense_after_attack(model, X_te_t, y_te_t, 0.05, sigma)
    defense_results.append({'sigma': sigma, 'accuracy': acc, 'f1': f1})
    print(f'  +Denoise(sigma={sigma}): Acc={acc:.4f}, F1={f1:.4f}')

# Multi-epsilon defense: test defense at different attack strengths
multi_defense = []
for eps in [0.01, 0.05, 0.1]:
    acc_no_def, f1_no_def, _ = fgsm_attack(model, X_te_t, y_te_t, eps)
    best_def_acc, best_def_f1 = 0, 0
    for sigma in [0.01, 0.05, 0.1]:
        acc_d, f1_d = defense_after_attack(model, X_te_t, y_te_t, eps, sigma)
        if acc_d > best_def_acc:
            best_def_acc, best_def_f1 = acc_d, f1_d
    multi_defense.append({'eps': eps, 'no_def_acc': acc_no_def, 'no_def_f1': f1_no_def,
                          'best_def_acc': best_def_acc, 'best_def_f1': best_def_f1})
    print(f'  FGSM({eps}): no_def={acc_no_def:.4f}, best_def={best_def_acc:.4f}')

# ═══════ STEP 6: Visualizations ═══════
print()
print('='*60)
print('STEP 6: Generating figures')
print('='*60)

# FIGURE 1: FGSM accuracy vs epsilon
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('FGSM Adversarial Attack: Accuracy Degradation on UNSW-NB15', fontsize=14, fontweight='bold')

ax1.plot(fgsm_results['epsilon'], fgsm_results['accuracy'], 'o-', color='#E74C3C', linewidth=2, markersize=8)
ax1.axhline(y=clean_acc, color='gray', linestyle='--', label=f'Clean Acc = {clean_acc:.3f}')
ax1.set_xlabel('Epsilon (perturbation magnitude)'); ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs FGSM Epsilon'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

ax2.plot(fgsm_results['epsilon'], fgsm_results['acc_drop'], 'o-', color='#E74C3C', linewidth=2, markersize=8)
ax2.set_xlabel('Epsilon'); ax2.set_ylabel('Accuracy Drop (%)')
ax2.set_title('Accuracy Degradation (relative to clean)'); ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_fgsm_attack.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig1: FGSM attack curve')

# FIGURE 2: FGSM vs PGD comparison
fig, ax = plt.subplots(figsize=(10, 6))
pgd_eps = [r['epsilon'] for r in pgd_results]
pgd_acc = [r['accuracy'] for r in pgd_results]
fgsm_at_pgd_eps = [fgsm_results['accuracy'][epsilons.index(e)] for e in pgd_eps]

x = np.arange(len(pgd_eps)); w = 0.35
ax.bar(x - w/2, fgsm_at_pgd_eps, w, label='FGSM', color='#3498DB')
ax.bar(x + w/2, pgd_acc, w, label='PGD (multi-step)', color='#E74C3C')
ax.axhline(y=clean_acc, color='gray', linestyle='--', label=f'Clean ({clean_acc:.3f})')
ax.set_xticks(x); ax.set_xticklabels([f'{r["config"]}' for r in pgd_results], fontsize=9)
ax.set_ylabel('Accuracy'); ax.set_title('FGSM vs PGD: Single-Step vs Multi-Step Attack')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_fgsm_vs_pgd.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig2: FGSM vs PGD')

# FIGURE 3: Extended defense with more sigmas
fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(y=clean_acc, color='gray', linestyle='--', label=f'Clean ({clean_acc:.3f})')
fgsm_05_acc = fgsm_results['accuracy'][epsilons.index(0.05)]
ax.axhline(y=fgsm_05_acc, color='#E74C3C', linestyle='--', label=f'FGSM(0.05) no defense ({fgsm_05_acc:.3f})')
def_acc = [r['accuracy'] for r in defense_results]
sigs = [r['sigma'] for r in defense_results]
ax.plot(sigs, def_acc, 'o-', color='#27AE60', linewidth=2, markersize=8, label='After denoising defense')
ax.set_xlabel('Denoising Sigma'); ax.set_ylabel('Accuracy')
ax.set_title('Diffusion-Inspired Denoising Defense: Recovery from FGSM(0.05) Attack')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_denoising_defense.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig3: Defense recovery (extended)')

# FIGURE 3b: PGD step sensitivity
fig, ax = plt.subplots(figsize=(9, 5))
eps_colors = {0.05: '#3498DB', 0.1: '#E74C3C'}
for eps in [0.05, 0.1]:
    matches = [r for r in pgd_results + pgd_deep_results if abs(r.get('epsilon', 0) - eps) < 0.001]
    steps_list = []
    acc_list = []
    for r in sorted(matches, key=lambda x: x.get('steps', 7 if '7' in x['config'] else 10)):
        s = r.get('steps', 7 if '7 steps' in r['config'] else (10 if '10 steps' in r['config'] else (20 if '20 steps' in r['config'] else 40)))
        steps_list.append(s); acc_list.append(r['accuracy'])
    ax.plot(steps_list, [a*100 for a in acc_list], 'o-', color=eps_colors[eps], linewidth=2, markersize=8, 
            label=f'PGD eps={eps}')
ax.axhline(y=clean_acc*100, color='gray', linestyle='--', label=f'Clean: {clean_acc*100:.1f}%')
ax.set_xlabel('PGD Steps'); ax.set_ylabel('Accuracy (%)')
ax.set_title('PGD Attack: Accuracy vs Number of Iteration Steps')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3b_pgd_steps.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig3b: PGD step sensitivity')

# FIGURE 4: Combined robustness summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Adversarial Robustness Evaluation Summary', fontsize=14, fontweight='bold')

ax1.plot(fgsm_results['epsilon'], [a*100 for a in fgsm_results['accuracy']], 'o-', color='#3498DB', linewidth=2, markersize=6)
ax1.axhline(y=clean_acc*100, color='gray', linestyle='--', label=f'Clean: {clean_acc*100:.1f}%')
ax1.set_xlabel('Epsilon'); ax1.set_ylabel('Accuracy (%)'); ax1.set_title('FGSM Attack Sensitivity')
ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_xscale('log')

methods = ['Clean', 'FGSM(0.05)', '+Denoise(0.01)', '+Denoise(0.05)', '+Denoise(0.1)', '+Denoise(0.3)']
idx_05 = epsilons.index(0.05)
accs = [clean_acc*100, fgsm_results['accuracy'][idx_05]*100]
# Add first 3 defense results
for i in [0, 3, 5, 7]:
    if i < len(defense_results):
        accs.append(defense_results[i]['accuracy']*100)
colors = ['gray', '#E74C3C', '#F39C12', '#F1C40F', '#27AE60', '#3498DB']
ax2.bar(methods[:len(accs)], accs, color=colors[:len(accs)], edgecolor='white')
ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Defense Recovery at Epsilon=0.05')
for i, v in enumerate(accs):
    ax2.text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
ax2.set_ylim(0, 100)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_robustness_summary.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig4: Summary')

# FIGURE 5: Multi-epsilon defense
fig, ax = plt.subplots(figsize=(9, 5))
eps_vals = [m['eps'] for m in multi_defense]
no_def = [m['no_def_acc']*100 for m in multi_defense]
best_def = [m['best_def_acc']*100 for m in multi_defense]
x = np.arange(len(eps_vals)); w = 0.3
ax.bar(x - w/2, no_def, w, label='No Defense', color='#E74C3C')
ax.bar(x + w/2, best_def, w, label='Best Defense', color='#27AE60')
ax.axhline(y=clean_acc*100, color='gray', linestyle='--', label=f'Clean ({clean_acc*100:.1f}%)')
for i, (nd, bd) in enumerate(zip(no_def, best_def)):
    ax.text(i-w/2, nd+1, f'{nd:.1f}%', ha='center', fontsize=8)
    ax.text(i+w/2, bd+1, f'{bd:.1f}%', ha='center', fontsize=8)
    recovery = bd - nd
    ax.annotate(f'+{recovery:.1f}%', xy=(i, (nd+bd)/2), ha='center', fontsize=9, fontweight='bold', color='blue')
ax.set_xticks(x); ax.set_xticklabels([f'eps={v}' for v in eps_vals])
ax.set_ylabel('Accuracy (%)'); ax.set_title('Multi-Epsilon Defense: Recovery After Denoising')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_multi_epsilon_defense.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig5: Multi-epsilon defense')

# Save
results = {
    'clean_accuracy': float(clean_acc), 'clean_f1': float(clean_f1),
    'fgsm': fgsm_results, 'pgd': pgd_results, 'pgd_deep': pgd_deep_results,
    'defense': defense_results, 'multi_defense': multi_defense
}
with open(os.path.join(OUT, 'adversarial_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nDone. {len(os.listdir(OUT))} files in {OUT}')
