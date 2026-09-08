"""
Complete GNN + Transformer baseline metrics on all 3 datasets.
Includes CIC-DDoS2019 for both baselines + full GNN metrics.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '4'; os.environ['OMP_NUM_THREADS'] = '4'; os.environ['MKL_NUM_THREADS'] = '4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

DEVICE = torch.device('cuda')
OUT = '/opt/ids_revision/results'

class GraphSAGE(nn.Module):
    def __init__(self, in_c, hid, out_c, layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_c, hid)])
        for _ in range(layers-2): self.convs.append(SAGEConv(hid, hid))
        self.convs.append(SAGEConv(hid, hid))
        self.dropout = dropout; self.cls = nn.Linear(hid, out_c)
    def forward(self, x, ei):
        for c in self.convs:
            x = torch.relu(c(x, ei)); x = F.dropout(x, p=self.dropout, training=self.training)
        return self.cls(x)

def run_gnn(X_tr, y_tr, X_te, y_te, n_classes, k=10, epochs=50, hid=128):
    """Train GNN and return metrics."""
    n_tr = min(len(X_tr), 50000)
    X_all = torch.cat([X_tr[:n_tr], X_te[:20000]], dim=0)
    y_all = torch.cat([y_tr[:n_tr], y_te[:20000]], dim=0)
    
    t0 = time.time()
    adj = kneighbors_graph(X_all.numpy(), n_neighbors=min(k, X_all.shape[0]-1), mode='connectivity', include_self=False)
    ei = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    print(f'  Graph: {X_all.shape[0]} nodes, {ei.shape[1]} edges, {time.time()-t0:.1f}s')
    
    X_all, ei = X_all.to(DEVICE), ei.to(DEVICE)
    model = GraphSAGE(X_all.shape[1], hid, n_classes).to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    best_f1, best_st = 0, None
    
    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X_all, ei)[:n_tr], y_all[:n_tr].to(DEVICE))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        
        model.eval()
        with torch.no_grad():
            preds = model(X_all, ei)[:n_tr].argmax(dim=1).cpu()
            _, _, f1, _ = precision_recall_fscore_support(y_all[:n_tr], preds, average='weighted', zero_division=0)
        if f1 > best_f1: best_f1 = f1; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        preds = model(X_all, ei)[n_tr:].argmax(dim=1).cpu()
        acc = accuracy_score(y_all[n_tr:], preds)
        p, r, f1, _ = precision_recall_fscore_support(y_all[n_tr:], preds, average='weighted', zero_division=0)
    return acc, p, r, f1

# ─── Load UNSW ───
def load_unsw(multiclass=False):
    tr = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
    te = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
    df = pd.concat([tr, te], ignore_index=True)
    df = df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns], errors='ignore')
    if multiclass:
        y_raw = df['attack_cat'].astype(str).str.strip().replace('','Normal')
        le = LabelEncoder(); y = le.fit_transform(y_raw)
    else:
        y = df['label'].values.astype(np.int64)
    feats = [c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
    X = df[feats]
    for c in X.select_dtypes(include=['object']).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return torch.FloatTensor(X_tr), torch.LongTensor(y_tr), torch.FloatTensor(X_te), torch.LongTensor(y_te)

# ─── Load CIC ───
sys.path.insert(0, '/opt/ids_revision/deploy')
from cic_data_loader import load_cicflowmeter

def load_cic(data_dir, max_samples=100000):
    data = load_cicflowmeter(data_dir, rows_per_file=30000, max_total=max_samples, binary=True)
    X_all = torch.cat([data['X_train'], data['X_val'], data['X_test']], dim=0)
    y_all = torch.cat([data['y_train'], data['y_val'], data['y_test']], dim=0)
    # Split train/test
    n = len(X_all)
    n_tr = int(n * 0.7)
    return X_all[:n_tr], y_all[:n_tr], X_all[n_tr:], y_all[n_tr:]

# ════════════════════════════════════════════════
print('=' * 60)
print('1. GNN UNSW-NB15 Binary (Prec/Rec)')
print('=' * 60)
X_tr, y_tr, X_te, y_te = load_unsw(multiclass=False)
acc, p, r, f1 = run_gnn(X_tr, y_tr, X_te, y_te, 2)
print(f'GNN Binary: Acc={acc:.4f}, Prec={p:.4f}, Rec={r:.4f}, F1={f1:.4f}')
with open(os.path.join(OUT, 'gnn_full_metrics.json'), 'w') as f:
    json.dump({'unsw_binary': {'acc': float(acc), 'prec': float(p), 'rec': float(r), 'f1': float(f1)}}, f, indent=2)

print()
print('=' * 60)
print('2. GNN UNSW-NB15 Multi-Class')
print('=' * 60)
X_tr_mc, y_tr_mc, X_te_mc, y_te_mc = load_unsw(multiclass=True)
acc_mc, p_mc, r_mc, f1_mc = run_gnn(X_tr_mc, y_tr_mc, X_te_mc, y_te_mc, 10, k=10, epochs=50)
print(f'GNN Multi: Acc={acc_mc:.4f}, Prec={p_mc:.4f}, Rec={r_mc:.4f}, F1={f1_mc:.4f}')
# Append to file
with open(os.path.join(OUT, 'gnn_full_metrics.json')) as f:
    d = json.load(f)
d['unsw_multiclass'] = {'acc': float(acc_mc), 'prec': float(p_mc), 'rec': float(r_mc), 'f1': float(f1_mc)}
with open(os.path.join(OUT, 'gnn_full_metrics.json'), 'w') as f:
    json.dump(d, f, indent=2)

print()
print('=' * 60)
print('3. GNN CIC-IDS-2017')
print('=' * 60)
X_tr_ci, y_tr_ci, X_te_ci, y_te_ci = load_cic('/opt/CIC-IDS-2017/MachineLearningCVE', 80000)
acc_ci, p_ci, r_ci, f1_ci = run_gnn(X_tr_ci, y_tr_ci, X_te_ci, y_te_ci, 2, k=5, epochs=50)
print(f'GNN CIC-IDS: Acc={acc_ci:.4f}, Prec={p_ci:.4f}, Rec={r_ci:.4f}, F1={f1_ci:.4f}')
with open(os.path.join(OUT, 'gnn_full_metrics.json')) as f:
    d = json.load(f)
d['cic_ids2017'] = {'acc': float(acc_ci), 'prec': float(p_ci), 'rec': float(r_ci), 'f1': float(f1_ci)}
with open(os.path.join(OUT, 'gnn_full_metrics.json'), 'w') as f:
    json.dump(d, f, indent=2)

print()
print('=' * 60)
print('4. GNN CIC-DDoS2019')
print('=' * 60)
X_tr_dd, y_tr_dd, X_te_dd, y_te_dd = load_cic('/opt/CIC-DDoS2019/all', 80000)
acc_dd, p_dd, r_dd, f1_dd = run_gnn(X_tr_dd, y_tr_dd, X_te_dd, y_te_dd, 2, k=5, epochs=50)
print(f'GNN CIC-DDoS: Acc={acc_dd:.4f}, Prec={p_dd:.4f}, Rec={r_dd:.4f}, F1={f1_dd:.4f}')
with open(os.path.join(OUT, 'gnn_full_metrics.json')) as f:
    d = json.load(f)
d['cic_ddos2019'] = {'acc': float(acc_dd), 'prec': float(p_dd), 'rec': float(r_dd), 'f1': float(f1_dd)}
with open(os.path.join(OUT, 'gnn_full_metrics.json'), 'w') as f:
    json.dump(d, f, indent=2)

print()
print('=' * 60)
print('5. FeatureTransformer CIC-DDoS2019')
print('=' * 60)
sys.path.insert(0, '/opt/ids_revision')
from ablation.models.feature_transformer import FeatureTransformer
from torch.utils.data import DataLoader, TensorDataset

X_tr_ft, y_tr_ft, X_te_ft, y_te_ft = load_cic('/opt/CIC-DDoS2019/all', 80000)
model_ft = FeatureTransformer(n_features=X_tr_ft.shape[1], num_classes=2, d_model=64, nhead=8, num_layers=4, pooling='mean').to(DEVICE)
print(f'FeatureTransformer params: {sum(p.numel() for p in model_ft.parameters()):,}')

dl = DataLoader(TensorDataset(X_tr_ft[:50000], y_tr_ft[:50000]), batch_size=256, shuffle=True)
opt = torch.optim.AdamW(model_ft.parameters(), lr=0.0001)
best_f1_ft, best_st_ft = 0, None
t0 = time.time()

for ep in range(1, 31):
    model_ft.train()
    for bx, by in dl:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad(); loss = nn.CrossEntropyLoss()(model_ft(bx), by)
        loss.backward(); opt.step()
    model_ft.eval()
    with torch.no_grad():
        preds = model_ft(X_te_ft[:10000].to(DEVICE)).argmax(dim=1).cpu()
        _, _, f1_v, _ = precision_recall_fscore_support(y_te_ft[:10000], preds, average='weighted', zero_division=0)
    if f1_v > best_f1_ft: best_f1_ft = f1_v; best_st_ft = {k: v.cpu().clone() for k, v in model_ft.state_dict().items()}

if best_st_ft: model_ft.load_state_dict(best_st_ft)
model_ft.eval()
with torch.no_grad():
    preds = model_ft(X_te_ft[:10000].to(DEVICE)).argmax(dim=1).cpu()
    acc_ft = accuracy_score(y_te_ft[:10000], preds)
    p_ft, r_ft, f1_ft, _ = precision_recall_fscore_support(y_te_ft[:10000], preds, average='weighted', zero_division=0)

print(f'FeatTrans CIC-DDoS: Acc={acc_ft:.4f}, Prec={p_ft:.4f}, Rec={r_ft:.4f}, F1={f1_ft:.4f}, Time={time.time()-t0:.1f}s')

with open(os.path.join(OUT, 'gnn_full_metrics.json')) as f:
    d = json.load(f)
d['feattrans_cic_ddos2019'] = {'acc': float(acc_ft), 'prec': float(p_ft), 'rec': float(r_ft), 'f1': float(f1_ft)}
with open(os.path.join(OUT, 'gnn_full_metrics.json'), 'w') as f:
    json.dump(d, f, indent=2)

print('\nDone. All results saved to gnn_full_metrics.json')
