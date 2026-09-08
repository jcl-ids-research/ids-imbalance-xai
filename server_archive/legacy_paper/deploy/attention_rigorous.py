"""
RIGOROUS attention visualization:
1. Train full model on UNSW-NB15 (50 epochs, save weights)
2. Load trained model
3. Extract attention from 15 samples per class (Normal/DoS/Fuzzers)
4. Average attention per class, generate statistical heatmaps
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, types, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '4'; os.environ['OMP_NUM_THREADS'] = '4'; os.environ['MKL_NUM_THREADS'] = '4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/opt/ids_revision')
from ablation.models.model_variants import create_variant, count_parameters

DEVICE = torch.device('cuda')
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

OUT = '/opt/ids_revision/results/attention_viz'
os.makedirs(OUT, exist_ok=True)
MODEL_PATH = os.path.join(OUT, 'full_model_trained.pt')

# ═══════════════════════════════════════════════════════
# STEP 1: Load and preprocess UNSW-NB15
# ═══════════════════════════════════════════════════════
print('='*60)
print('STEP 1: Loading UNSW-NB15')
print('='*60)

tr = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
te = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
df = pd.concat([tr, te], ignore_index=True)
df = df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns], errors='ignore')

# Save attack categories before encoding
attack_cats = df['attack_cat'].astype(str).str.strip().replace('','Normal').values
labels = df['label'].values.astype(np.int64)

feats = [c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
X = df[feats]
for c in X.select_dtypes(include=['object']).columns:
    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
X = StandardScaler().fit_transform(X)

# Split and keep indices for later attention extraction
indices = np.arange(len(X))
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, labels, indices, test_size=0.2, random_state=SEED, stratify=labels)
X_tr, X_vl, y_tr, y_vl = train_test_split(
    X_tr, y_tr, test_size=0.125, random_state=SEED, stratify=y_tr)

print(f'Train: {X_tr.shape}, Val: {X_vl.shape}, Test: {X_te.shape}')
print(f'Class dist: {np.bincount(y_tr)}')

X_tr_t = torch.FloatTensor(X_tr); y_tr_t = torch.LongTensor(y_tr)
X_vl_t = torch.FloatTensor(X_vl); y_vl_t = torch.LongTensor(y_vl)
X_te_t = torch.FloatTensor(X_te); y_te_t = torch.LongTensor(y_te)

# ═══════════════════════════════════════════════════════
# STEP 2: Train full model
# ═══════════════════════════════════════════════════════
print()
print('='*60)
print('STEP 2: Training Full Model (50 epochs)')
print('='*60)

model = create_variant('full', input_dim=42, num_classes=2,
                       d_model=128, nhead=8, num_layers=4, n_views=3,
                       fusion_method='concat', dim_feedforward=512, dropout=0.1).to(DEVICE)
print(f'Params: {count_parameters(model):,}')

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True)
val_loader = DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=512)

crit = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

best_f1, best_state = 0, None
t0 = time.time()

for ep in range(1, 51):
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad(); loss = crit(model(bx), by)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sch.step()
    
    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            preds = model(bx.to(DEVICE)).argmax(dim=1)
            ap.extend(preds.cpu().numpy()); al.extend(by.numpy())
    _, _, f1, _ = precision_recall_fscore_support(al, ap, average='weighted', zero_division=0)
    
    if f1 > best_f1:
        best_f1 = f1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if ep % 10 == 0:
        acc = accuracy_score(al, ap)
        print(f'  Epoch {ep:2d}/50  val_acc={acc:.4f}  val_f1={f1:.4f}')

# Restore best and evaluate
model.load_state_dict(best_state)
model.eval()
ap, al = [], []
with torch.no_grad():
    for bx, by in DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=512):
        preds = model(bx.to(DEVICE)).argmax(dim=1)
        ap.extend(preds.cpu().numpy()); al.extend(by.numpy())

acc = accuracy_score(al, ap)
p, r, f1, _ = precision_recall_fscore_support(al, ap, average='weighted', zero_division=0)
print(f'\nTest: Acc={acc:.4f}, F1={f1:.4f}, Time={time.time()-t0:.1f}s')

# Save weights
torch.save(best_state, MODEL_PATH)
print(f'Model saved: {MODEL_PATH}')

# ═══════════════════════════════════════════════════════
# STEP 3: Extract attention weights from trained model
# ═══════════════════════════════════════════════════════
print()
print('='*60)
print('STEP 3: Extracting Attention Weights')
print('='*60)

# Select samples: 15 each for Normal, DoS, Fuzzers
test_cats = attack_cats[idx_te]
test_labels = labels[idx_te]

samples_per_class = 15
selected = {'Normal': [], 'DoS': [], 'Fuzzers': []}

for cat in selected.keys():
    candidates = np.where(test_cats == cat)[0]
    # For DoS/Fuzzers, prefer correctly classified attack samples (label=1)
    if cat != 'Normal':
        attack_candidates = [i for i in candidates if test_labels[i] == 1]
        if len(attack_candidates) >= samples_per_class:
            candidates = np.array(attack_candidates)
    selected[cat] = np.random.RandomState(SEED).choice(candidates, min(samples_per_class, len(candidates)), replace=False).tolist()

for cat, idxs in selected.items():
    print(f'{cat}: {len(idxs)} samples')

# Hook attention layers
attention_storage = {}  # {view_name: {layer_name: List of attention matrices}}

for view_idx, view_transformer in enumerate(model.encoder.view_transformers):
    view_name = f'View_{view_idx+1}'
    attention_storage[view_name] = {}
    
    for layer_idx, layer in enumerate(view_transformer.blocks):
        layer_name = f'Layer_{layer_idx+1}'
        attention_storage[view_name][layer_name] = []
        
        original_forward = layer.self_attn.forward
        
        def make_hook(vn, ln):
            def hook(module, input, output):
                pass
            def patched(self, query, key, value, **kw):
                kw['need_weights'] = True; kw['average_attn_weights'] = True
                out, w = original_forward(query, key, value, **kw)
                attention_storage[vn][ln].append(w.detach().cpu().numpy()[0])
                return out, w
            return patched
        
        layer.self_attn.forward = types.MethodType(make_hook(view_name, layer_name), layer.self_attn)

# Forward all selected samples
for cat, idxs in selected.items():
    for idx in idxs:
        x = X_te_t[idx:idx+1].to(DEVICE)
        with torch.no_grad():
            _ = model(x)

# ═══════════════════════════════════════════════════════
# STEP 4: Generate visualizations
# ═══════════════════════════════════════════════════════
print()
print('='*60)
print('STEP 4: Generating Attention Visualizations')
print('='*60)

# 4a: Per-class average attention heatmaps (last layer, each view)
for cat in ['Normal', 'DoS', 'Fuzzers']:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Average Multi-View Attention — {cat} ({len(selected[cat])} samples)', fontsize=14, fontweight='bold')
    
    for vi, view_name in enumerate(['View_1', 'View_2', 'View_3']):
        ax = axes[vi]
        layer_keys = sorted(attention_storage[view_name].keys())
        last_layer = layer_keys[-1]
        
        # Get the attention matrices for this class
        start_idx = sum(len(selected[k]) for k in ['Normal','DoS','Fuzzers'] if list(selected.keys()).index(k) < list(selected.keys()).index(cat))
        class_attns = attention_storage[view_name][last_layer][start_idx * len(selected[cat]) : (start_idx + len(selected[cat])) * len(selected[cat])]
        
        # Hmm, this indexing is wrong because attention_storage collects ALL samples sequentially
        # Let me recalculate
        n_before = sum(len(selected[k]) for k in ['Normal','DoS','Fuzzers'] if ['Normal','DoS','Fuzzers'].index(k) < ['Normal','DoS','Fuzzers'].index(cat))
        n_this = len(selected[cat])
        
        # All attention matrices are stored in order: Normal(15) then DoS(15) then Fuzzers(15)
        # Each sample contributes one matrix per (view, layer) combo
        class_start = n_before
        class_end = n_before + n_this
        class_attns = attention_storage[view_name][last_layer][class_start:class_end]
        
        if class_attns:
            avg_attn = np.mean(class_attns, axis=0)
            im = ax.imshow(avg_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=avg_attn.max())
            nf = avg_attn.shape[0]
            ax.set_xticks(range(nf)); ax.set_yticks(range(nf))
            ax.set_xticklabels([f'F{i+1}' for i in range(nf)], fontsize=5, rotation=90)
            ax.set_yticklabels([f'F{i+1}' for i in range(nf)], fontsize=5)
            ax.set_title(f'{view_name.replace(chr(95),chr(32))}\n({nf} features, {last_layer})', fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    
    plt.tight_layout()
    path = os.path.join(OUT, f'attention_avg_{cat}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] {path}')

# 4b: Cross-class attention comparison (bar chart of top attended features)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Attention Distribution Across Attack Types', fontsize=14, fontweight='bold')

for vi, view_name in enumerate(['View_1', 'View_2', 'View_3']):
    ax = axes[vi]
    layer_keys = sorted(attention_storage[view_name].keys())
    last_layer = layer_keys[-1]
    
    all_attns = attention_storage[view_name][last_layer]
    n_per_class = [len(selected[k]) for k in ['Normal','DoS','Fuzzers']]
    
    n0, n1 = n_per_class[0], n_per_class[0] + n_per_class[1]
    normal_avg = np.mean(all_attns[:n0], axis=0).mean(axis=0)  # avg attention per feature
    dos_avg = np.mean(all_attns[n0:n1], axis=0).mean(axis=0)
    fuzzers_avg = np.mean(all_attns[n1:], axis=0).mean(axis=0)
    
    nf = len(normal_avg)
    x = np.arange(nf); w = 0.25
    ax.bar(x-w, normal_avg, w, label='Normal', alpha=0.7)
    ax.bar(x, dos_avg, w, label='DoS', alpha=0.7)
    ax.bar(x+w, fuzzers_avg, w, label='Fuzzers', alpha=0.7)
    ax.set_title(f'{view_name.replace(chr(95),chr(32))} ({nf} features)', fontsize=11)
    ax.set_xlabel('Feature index'); ax.set_ylabel('Avg Attention')
    ax.legend(fontsize=8)

plt.tight_layout()
path = os.path.join(OUT, 'attention_class_comparison.png')
fig.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f'[OK] {path}')

# 4c: Attention concentration curve
fig, ax = plt.subplots(figsize=(10, 5))
for cat, color in [('Normal','blue'),('DoS','red'),('Fuzzers','green')]:
    n_before = sum(len(selected[k]) for k in ['Normal','DoS','Fuzzers'] if ['Normal','DoS','Fuzzers'].index(k) < ['Normal','DoS','Fuzzers'].index(cat))
    n_this = len(selected[cat])
    
    all_view_avg = []
    for view_name in ['View_1', 'View_2', 'View_3']:
        layer_keys = sorted(attention_storage[view_name].keys())
        last_layer = layer_keys[-1]
        class_attns = attention_storage[view_name][last_layer][n_before:n_before+n_this]
        for mat in class_attns:
            all_view_avg.extend(mat.mean(axis=0))
    
    all_view_avg = sorted(all_view_avg, reverse=True)
    cumsum = np.cumsum(all_view_avg) / np.sum(all_view_avg)
    ax.plot(np.arange(1, len(cumsum)+1)/len(cumsum)*100, cumsum*100, color=color, linewidth=2, label=f'{cat} ({n_this} samples)')

ax.set_xlabel('Cumulative Features (%)'); ax.set_ylabel('Cumulative Attention (%)')
ax.set_title('Attention Concentration Curve: How many features capture 80% of attention?')
ax.legend(); ax.grid(True, alpha=0.3)
ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
path = os.path.join(OUT, 'attention_concentration.png')
fig.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f'[OK] {path}')

print(f'\nDone. {len(os.listdir(OUT))} files in {OUT}')
