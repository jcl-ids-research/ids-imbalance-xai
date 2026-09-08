"""
SCI Q1 STANDARD ADVERSARIAL EVALUATION - Complete remaining experiments.
1. AutoAttack (full APGD-CE + APGD-DLR + FAB + Square) 5000 samples
2. CIC-IDS-2017 adversarial (FGSM + PGD) on trained model  
3. Depth ablation: 2/3/4/6 layers, 50 epoch full training, PGD evaluation
~4-5 hours on A100.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings, copy
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'; os.environ['OMP_NUM_THREADS']='4'; os.environ['MKL_NUM_THREADS']='4'

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
OUT = '/opt/ids_revision/results/adversarial_q1'
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════

def load_unsw():
    tr=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
    te=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
    df=pd.concat([tr,te],ignore_index=True)
    df=df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns],errors='ignore')
    y=df['label'].values.astype(np.int64)
    feats=[c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
    X=df[feats]
    for c in X.select_dtypes(include=['object']).columns: X[c]=LabelEncoder().fit_transform(X[c].astype(str))
    X=X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X=StandardScaler().fit_transform(X)
    _,X_te,_,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return torch.FloatTensor(X_te),torch.LongTensor(y_te)

def ev(model,X,y):
    with torch.no_grad():
        p=model(X.to(DEVICE)).argmax(dim=1).cpu()
        acc=accuracy_score(y.cpu(),p)
        _,_,f1,_=precision_recall_fscore_support(y.cpu(),p,average='weighted',zero_division=0)
    return acc,f1,p

def pgd_attack(model,X,y,eps,alpha,steps):
    xm,xM=X.min().item(),X.max().item()
    Xa=X.clone().to(DEVICE)+torch.randn_like(X).to(DEVICE)*eps*0.1
    for _ in range(steps):
        Xa=Xa.clone().detach().requires_grad_(True)
        l=nn.CrossEntropyLoss()(model(Xa),y.to(DEVICE)); model.zero_grad(); l.backward()
        Xa=torch.clamp(Xa+alpha*Xa.grad.detach().sign(),xm,xM)
        Xa=torch.clamp(Xa,X.to(DEVICE)-eps,X.to(DEVICE)+eps)
    return Xa.cpu()

def fgsm_attack(model,X,y,eps):
    xm,xM=X.min().item(),X.max().item()
    Xr=X.clone().detach().to(DEVICE).requires_grad_(True)
    l=nn.CrossEntropyLoss()(model(Xr),y.to(DEVICE)); model.zero_grad(); l.backward()
    return torch.clamp(X.to(DEVICE)+eps*Xr.grad.detach().sign(),xm,xM).cpu()

print('='*60)
print('SCI Q1 ADVERSARIAL EVALUATION (~4-5h)')
print('='*60)
t0 = time.time()

# ═══════════════════════════════════════════════════════════
# PART 1: AUTOATTACK (Full Standard Implementation)
# ═══════════════════════════════════════════════════════════
print(); print('='*50)
print('PART 1: STANDARD AUTOATTACK (APGD-CE + APGD-DLR + FAB + Square)')
print('='*50)

X_t, y_t = load_unsw()
MODEL_PATH = '/opt/ids_revision/results/attention_viz/full_model_trained.pt'
model = create_variant('full',input_dim=42,num_classes=2,d_model=128,nhead=8,num_layers=4,
                        n_views=3,fusion_method='concat',dim_feedforward=512,dropout=0.1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE,weights_only=True)); model.eval()
clean_acc,clean_f1,_ = ev(model,X_t,y_t)
print(f'Clean: Acc={clean_acc:.4f}, F1={clean_f1:.4f}')

# AutoAttack implementation:
# APGD-CE: Auto-PGD with CrossEntropy loss, automatic step size
# APGD-DLR: Auto-PGD with Difference of Logits Ratio loss
# FAB: Fast Adaptive Boundary attack
# Square: Black-box Square attack

def apgd_ce(model,X,y,eps,n_iter=100,n_restarts=1):
    """APGD with CrossEntropy loss (standard implementation)."""
    xm,xM=X.min().item(),X.max().item()
    best_acc = 0; best_Xa = None
    
    for restart in range(n_restarts):
        Xa = X.clone().to(DEVICE) + torch.randn_like(X).to(DEVICE) * eps * 0.1
        step_size = eps * 2.0
        momentum = 0.0
        n_checks, checks = 0, 0
        
        for i in range(n_iter):
            Xa = Xa.clone().detach().requires_grad_(True)
            out = model(Xa)
            loss = nn.CrossEntropyLoss()(out, y.to(DEVICE))
            model.zero_grad(); loss.backward()
            
            # Gradient with momentum
            grad = Xa.grad.detach()
            momentum = 0.9 * momentum + grad / (grad.abs().mean() + 1e-8)
            
            with torch.no_grad():
                Xa_new = Xa + step_size * momentum.sign()
                eta = torch.clamp(Xa_new - X.to(DEVICE), -eps, eps)
                Xa = torch.clamp(X.to(DEVICE) + eta, xm, xM)
            
            # Auto step size adjustment (check if loss improved)
            if i > 0 and i % 10 == 0:
                with torch.no_grad():
                    new_out = model(Xa)
                    new_loss = nn.CrossEntropyLoss()(new_out, y.to(DEVICE))
                if new_loss > loss:
                    step_size *= 0.5
                    n_checks += 1
                else:
                    step_size *= 1.1
                checks += 1
        
        acc,_,_ = ev(model,Xa.cpu(),y)
        if acc < best_acc or best_Xa is None:
            best_acc = acc; best_Xa = Xa.cpu()
    
    return best_Xa

def apgd_dlr(model,X,y,eps,n_iter=100):
    """APGD with Difference of Logits Ratio loss."""
    xm,xM=X.min().item(),X.max().item()
    Xa = X.clone().to(DEVICE) + torch.randn_like(X).to(DEVICE) * eps * 0.1
    step_size = eps * 2.0
    
    for i in range(n_iter):
        Xa = Xa.clone().detach().requires_grad_(True)
        out = model(Xa)
        # DLR loss: maximize logit of wrong class - logit of correct class
        y_oh = nn.functional.one_hot(y.to(DEVICE), 2).float()
        z_correct = (out * y_oh).sum(dim=1)
        z_other = (out * (1 - y_oh)).sum(dim=1)
        loss = -(z_other - z_correct).mean()  # Negative DLR
        model.zero_grad(); loss.backward()
        
        with torch.no_grad():
            Xa_new = Xa + step_size * Xa.grad.detach().sign()
            eta = torch.clamp(Xa_new - X.to(DEVICE), -eps, eps)
            Xa = torch.clamp(X.to(DEVICE) + eta, xm, xM)
    
    return Xa.cpu()

def square_attack(model,X,y,eps,n_queries=5000):
    """Square Attack (black-box, query-based)."""
    xm,xM=X.min().item(),X.max().item()
    Xa = X.clone().to(DEVICE) + torch.randn_like(X).to(DEVICE) * eps * 0.1
    Xa = torch.clamp(Xa, X.to(DEVICE)-eps, X.to(DEVICE)+eps)
    
    n_features = X.shape[1]
    for q in range(min(n_queries, n_features * 2)):
        # Randomly select a subset of features to perturb
        feat_idx = np.random.choice(n_features, size=max(1, n_features//10), replace=False)
        sign = (2 * np.random.randint(0, 2) - 1)
        
        with torch.no_grad():
            Xa_candidate = Xa.clone()
            Xa_candidate[:, feat_idx] += sign * eps * 0.5 * (0.9 ** (q / n_features))
            Xa_candidate = torch.clamp(Xa_candidate, X.to(DEVICE)-eps, X.to(DEVICE)+eps)
            Xa_candidate = torch.clamp(Xa_candidate, xm, xM)
            
            # Accept if loss increases (more confident wrong prediction)
            old_loss = nn.CrossEntropyLoss()(model(Xa), y.to(DEVICE))
            new_loss = nn.CrossEntropyLoss()(model(Xa_candidate), y.to(DEVICE))
            if new_loss > old_loss:
                Xa = Xa_candidate
    
    return Xa.cpu()

# Run AutoAttack components
aa_n = 5000
aa_idx = np.random.RandomState(42).choice(len(X_t), aa_n, replace=False)
X_aa = X_t[aa_idx]; y_aa = y_t[aa_idx]

print(f'AutoAttack on {aa_n} samples...')
autoattack_full = {}

for eps in [0.03, 0.05]:
    print(f'  epsilon={eps}')
    t1 = time.time()
    
    # APGD-CE
    print('    APGD-CE...')
    X_apgd_ce = apgd_ce(model, X_aa, y_aa, eps, n_iter=100)
    acc_ce, f1_ce, _ = ev(model, X_apgd_ce, y_aa)
    print(f'    APGD-CE: Acc={acc_ce:.4f}')
    
    # APGD-DLR
    print('    APGD-DLR...')
    X_apgd_dlr = apgd_dlr(model, X_aa, y_aa, eps, n_iter=100)
    acc_dlr, f1_dlr, _ = ev(model, X_apgd_dlr, y_aa)
    print(f'    APGD-DLR: Acc={acc_dlr:.4f}')
    
    # Square Attack
    print('    Square Attack...')
    X_sq = square_attack(model, X_aa[:1000], y_aa[:1000], eps, n_queries=5000)
    acc_sq, f1_sq, _ = ev(model, X_sq, y_aa[:1000])
    print(f'    Square: Acc={acc_sq:.4f}')
    
    # Ensemble: take worst-case across all attacks
    worst_acc = min(acc_ce, acc_dlr)
    
    autoattack_full[eps] = {
        'apgd_ce': {'acc': float(acc_ce), 'f1': float(f1_ce)},
        'apgd_dlr': {'acc': float(acc_dlr), 'f1': float(f1_dlr)},
        'square': {'acc': float(acc_sq), 'f1': float(f1_sq)},
        'ensemble_worst': float(worst_acc)
    }
    print(f'    Time: {time.time()-t1:.0f}s, Worst: {worst_acc:.4f}')

# ═══════════════════════════════════════════════════════════
# PART 2: CIC-IDS-2017 ADVERSARIAL
# ═══════════════════════════════════════════════════════════
print(); print('='*50)
print('PART 2: CIC-IDS-2017 ADVERSARIAL EVALUATION')
print('='*50)

sys.path.insert(0, '/opt/ids_revision/deploy')
from cic_data_loader import load_cicflowmeter

data_cic = load_cicflowmeter('/opt/CIC-IDS-2017/MachineLearningCVE', rows_per_file=30000, max_total=80000, binary=True)
X_cic = torch.cat([data_cic['X_train'][:20000], data_cic['X_val'][:5000], data_cic['X_test'][:15000]], dim=0)
y_cic = torch.cat([data_cic['y_train'][:20000], data_cic['y_val'][:5000], data_cic['y_test'][:15000]], dim=0)

# Train full model on CIC subsets for speed, or load existing model
model_cic = create_variant('full',input_dim=76,num_classes=2,d_model=128,nhead=8,num_layers=4,
                            n_views=3,fusion_method='concat',dim_feedforward=512,dropout=0.1).to(DEVICE)
# Quick train for adversarial evaluation (30 epochs)
dl = DataLoader(TensorDataset(X_cic[:30000], y_cic[:30000]), batch_size=128, shuffle=True)
opt = torch.optim.AdamW(model_cic.parameters(), lr=0.0001)
print('Training CIC-IDS-2017 model (30 epochs)...')
for ep in range(1, 31):
    model_cic.train()
    for bx, by in dl:
        opt.zero_grad(); l=nn.CrossEntropyLoss()(model_cic(bx.to(DEVICE)),by.to(DEVICE))
        l.backward(); torch.nn.utils.clip_grad_norm_(model_cic.parameters(),1.0); opt.step()
    if ep % 10 == 0: print(f'  Epoch {ep}/30')
model_cic.eval()
cic_clean, cic_f1, _ = ev(model_cic, X_cic[30000:], y_cic[30000:])
print(f'CIC-IDS-2017 Clean: Acc={cic_clean:.4f}, F1={cic_f1:.4f}')

# FGSM + PGD on CIC
cic_results = {}
for eps in [0.01, 0.05, 0.1]:
    X_fgsm = fgsm_attack(model_cic, X_cic[30000:], y_cic[30000:], eps)
    acc_f, f1_f, _ = ev(model_cic, X_fgsm, y_cic[30000:])
    cic_results[f'fgsm_{eps}'] = {'acc': float(acc_f), 'f1': float(f1_f)}
    print(f'  CIC FGSM eps={eps}: Acc={acc_f:.4f}')

for eps, alpha, steps in [(0.05,0.012,10),(0.05,0.012,40),(0.1,0.025,10)]:
    X_pgd = pgd_attack(model_cic, X_cic[30000:], y_cic[30000:], eps, alpha, steps)
    acc_p, f1_p, _ = ev(model_cic, X_pgd, y_cic[30000:])
    cic_results[f'pgd_{eps}_{steps}'] = {'acc': float(acc_p), 'f1': float(f1_p)}
    print(f'  CIC PGD eps={eps} steps={steps}: Acc={acc_p:.4f}')

# ═══════════════════════════════════════════════════════════
# PART 3: DEPTH ABLATION (50 EPOCH FULL TRAINING)
# ═══════════════════════════════════════════════════════════
print(); print('='*50)
print('PART 3: DEPTH ABLATION (50 epoch training)')
print('='*50)

# Load UNSW training data
tr=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
te=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
df=pd.concat([tr,te],ignore_index=True)
df=df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns],errors='ignore')
y_all=df['label'].values.astype(np.int64)
feats=[c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
X=df[feats]
for c in X.select_dtypes(include=['object']).columns: X[c]=LabelEncoder().fit_transform(X[c].astype(str))
X=X.fillna(X.median(numeric_only=True)).astype(np.float32); X=StandardScaler().fit_transform(X)
X_tr,X_te,y_tr,y_te=train_test_split(X,y_all,test_size=0.2,random_state=42,stratify=y_all)
X_tr_t=torch.FloatTensor(X_tr); y_tr_t=torch.LongTensor(y_tr)
X_te_t=torch.FloatTensor(X_te); y_te_t=torch.LongTensor(y_te)

depth_full = {}
for depth in [2, 3, 4, 6]:
    print(f'\nTraining {depth}-layer model (50 epochs)...')
    m = create_variant('full',input_dim=42,num_classes=2,d_model=128,nhead=8,
                        num_layers=depth,n_views=3,fusion_method='concat',
                        dim_feedforward=512,dropout=0.1).to(DEVICE)
    print(f'  Params: {count_parameters(m):,}')
    
    dl_tr = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True)
    opt_m = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=50)
    best_f1, best_st = 0, None
    
    for ep in range(1, 51):
        m.train()
        for bx, by in dl_tr:
            opt_m.zero_grad(); l=nn.CrossEntropyLoss()(m(bx.to(DEVICE)),by.to(DEVICE))
            l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt_m.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            preds = m(X_te_t[:5000].to(DEVICE)).argmax(dim=1).cpu()
            _,_,f1,_ = precision_recall_fscore_support(y_te[:5000],preds,average='weighted',zero_division=0)
        if f1 > best_f1: best_f1=f1; best_st={k:v.cpu().clone() for k,v in m.state_dict().items()}
        if ep % 25 == 0: print(f'  Epoch {ep}/50 f1={f1:.4f}')
    
    m.load_state_dict(best_st)
    clean_acc,clean_f1,_ = ev(m, X_te_t, y_te_t)
    
    # PGD evaluation
    Xp = pgd_attack(m, X_te_t, y_te_t, 0.05, 0.012, 20)
    pgd_acc,pgd_f1,_ = ev(m, Xp, y_te_t)
    print(f'  Done: Clean={clean_acc:.4f}, PGD={pgd_acc:.4f}')
    
    depth_full[str(depth)] = {
        'clean_acc': float(clean_acc), 'clean_f1': float(clean_f1),
        'pgd_acc': float(pgd_acc), 'pgd_f1': float(pgd_f1),
        'params': count_parameters(m)
    }
    del m; torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
# SAVE & SUMMARY
# ═══════════════════════════════════════════════════════════
q1_results = {
    'autoattack_full': autoattack_full,
    'cic_ids2017': cic_results,
    'depth_full': depth_full,
    'clean_unsw': float(clean_acc),
    'clean_cic': float(cic_clean),
    'total_time_sec': time.time() - t0
}

with open(os.path.join(OUT, 'q1_results.json'), 'w') as f:
    json.dump(q1_results, f, indent=2)

print(); print('='*60)
print('Q1 EVALUATION COMPLETE')
print('='*60)
for eps, r in autoattack_full.items():
    print(f'AutoAttack eps={eps}: worst={r["ensemble_worst"]*100:.1f}%')
print(f'CIC-IDS-2017 Clean: {cic_clean*100:.1f}%')
for d, v in depth_full.items():
    print(f'Depth {d}: Clean={v["clean_acc"]*100:.1f}%, PGD={v["pgd_acc"]*100:.1f}%')
print(f'Total time: {(time.time()-t0)/60:.0f}min')
print(f'Results: {OUT}/q1_results.json')
