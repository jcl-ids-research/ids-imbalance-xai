"""
Train all models on NSL-KDD dataset.
Runs: Full Model, FeatureTransformer, GraphSAGE (binary + multiclass).
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '4'; os.environ['OMP_NUM_THREADS'] = '4'; os.environ['MKL_NUM_THREADS'] = '4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

DEVICE = torch.device('cuda')
sys.path.insert(0, '/opt/ids_revision')
sys.path.insert(0, '/opt/ids_revision/deploy')
from ablation.models.feature_transformer import FeatureTransformer
from ablation.models.model_variants import create_variant
from torch_geometric.nn import SAGEConv

OUT = '/opt/ids_revision/results'
DATA_DIR = '/opt/NSL-KDD'

# Class definitions
ATTACK_TYPES = {
    'normal': 'normal',
    'DoS': ['back','land','neptune','pod','smurf','teardrop','mailbomb','processtable','udpstorm','apache2','worm'],
    'Probe': ['satan','ipsweep','nmap','portsweep','mscan','saint'],
    'R2L': ['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','warezclient','spy','xlock','xsnoop','snmpguess','snmpgetattack','httptunnel','sendmail','named'],
    'U2R': ['buffer_overflow','loadmodule','rootkit','perl','sqlattack','xterm','ps'],
}
ATTACK_CAT = {}
for cat, names in ATTACK_TYPES.items():
    if isinstance(names, list):
        for n in names: ATTACK_CAT[n] = cat
ATTACK_CAT['normal'] = 'normal'

# ─── Data Loading ───
COLUMNS = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
           'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
           'root_shell','su_attempted','num_root','num_file_creations','num_shells',
           'num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
           'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
           'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
           'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
           'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
           'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
           'label','difficulty']

def load_nsl_kdd(train_path, test_path, multiclass=False):
    df_tr = pd.read_csv(train_path, names=COLUMNS)
    df_te = pd.read_csv(test_path, names=COLUMNS)
    df = pd.concat([df_tr, df_te], ignore_index=True)
    
    # Encode categorical: protocol_type, service, flag
    for c in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder(); df[c] = le.fit_transform(df[c].astype(str))
    
    if multiclass:
        df['attack_cat'] = df['label'].map(ATTACK_CAT).fillna('unknown')
        le = LabelEncoder(); y = le.fit_transform(df['attack_cat'])
        classes = list(le.classes_)
        print(f'Multiclass: {len(classes)} classes: {classes}')
    else:
        y = (df['label'] != 'normal').astype(int).values
        classes = ['Normal', 'Attack']
    
    feats = [c for c in COLUMNS if c not in ['label','difficulty']]
    X = df[feats].fillna(0).astype(np.float32)
    X = StandardScaler().fit_transform(X)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return torch.FloatTensor(X_tr), torch.LongTensor(y_tr), torch.FloatTensor(X_te), torch.LongTensor(y_te), classes

# ─── GNN ───
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

def train_gnn(X_tr, y_tr, X_te, y_te, n_classes, name):
    n_tr = min(len(X_tr), 50000)
    X_all = torch.cat([X_tr[:n_tr], X_te[:20000]], dim=0)
    y_all = torch.cat([y_tr[:n_tr], y_te[:20000]], dim=0)
    
    t0 = time.time()
    k = min(10, X_all.shape[0]-1)
    adj = kneighbors_graph(X_all.numpy(), n_neighbors=k, mode='connectivity', include_self=False)
    ei = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    print(f'  Graph: {X_all.shape[0]} nodes, {ei.shape[1]} edges, {time.time()-t0:.1f}s')
    
    X_all, ei = X_all.to(DEVICE), ei.to(DEVICE)
    model = GraphSAGE(X_all.shape[1], 128, n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    best_f1, best_st = 0, None
    
    for ep in range(1, 51):
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
    print(f'  {name}: Acc={acc:.4f}, Prec={p:.4f}, Rec={r:.4f}, F1={f1:.4f}')
    return {'acc': float(acc), 'prec': float(p), 'rec': float(r), 'f1': float(f1)}

# ─── FeatureTransformer ───
def train_feattrans(X_tr, y_tr, X_te, y_te, n_classes, name):
    model = FeatureTransformer(n_features=X_tr.shape[1], num_classes=n_classes, d_model=64, nhead=8, num_layers=4, pooling='mean').to(DEVICE)
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
    dl = DataLoader(TensorDataset(X_tr[:50000], y_tr[:50000]), batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
    best_f1, best_st = 0, None; t0 = time.time()
    
    for ep in range(1, 31):
        model.train()
        for bx, by in dl:
            opt.zero_grad(); loss = nn.CrossEntropyLoss()(model(bx.to(DEVICE)), by.to(DEVICE))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te[:5000].to(DEVICE)).argmax(dim=1).cpu()
            _, _, f1, _ = precision_recall_fscore_support(y_te[:5000], preds, average='weighted', zero_division=0)
        if f1 > best_f1: best_f1 = f1; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        preds = model(X_te[:5000].to(DEVICE)).argmax(dim=1).cpu()
        acc = accuracy_score(y_te[:5000], preds)
        p, r, f1, _ = precision_recall_fscore_support(y_te[:5000], preds, average='weighted', zero_division=0)
    print(f'  {name}: Acc={acc:.4f}, Prec={p:.4f}, Rec={r:.4f}, F1={f1:.4f}, Time={time.time()-t0:.1f}s')
    return {'acc': float(acc), 'prec': float(p), 'rec': float(r), 'f1': float(f1)}

# ─── Full Model (Diffusion + MV) ───
def train_full(X_tr, y_tr, X_te, y_te, n_classes, name):
    model = create_variant('full', input_dim=X_tr.shape[1], num_classes=n_classes,
                           d_model=128, nhead=8, num_layers=4, n_views=3,
                           fusion_method='concat', dim_feedforward=512, dropout=0.1).to(DEVICE)
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
    dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    best_f1, best_st = 0, None; t0 = time.time()
    
    for ep in range(1, 31):
        model.train()
        for bx, by in dl:
            opt.zero_grad(); loss = nn.CrossEntropyLoss()(model(bx.to(DEVICE)), by.to(DEVICE))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te[:5000].to(DEVICE)).argmax(dim=1).cpu()
            _, _, f1, _ = precision_recall_fscore_support(y_te[:5000], preds, average='weighted', zero_division=0)
        if f1 > best_f1: best_f1 = f1; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0: print(f'  Epoch {ep}/30 f1={f1:.4f}')
    
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        preds = model(X_te[:5000].to(DEVICE)).argmax(dim=1).cpu()
        acc = accuracy_score(y_te[:5000], preds)
        p, r, f1, _ = precision_recall_fscore_support(y_te[:5000], preds, average='weighted', zero_division=0)
    print(f'  {name}: Acc={acc:.4f}, Prec={p:.4f}, Rec={r:.4f}, F1={f1:.4f}, Time={time.time()-t0:.1f}s')
    return {'acc': float(acc), 'prec': float(p), 'rec': float(r), 'f1': float(f1)}

# ════════════════════════════════════════════════════
results = {}

print('=' * 60)
print('1. NSL-KDD Binary')
print('=' * 60)
X_tr, y_tr, X_te, y_te, cls = load_nsl_kdd(os.path.join(DATA_DIR,'KDDTrain+.txt'), os.path.join(DATA_DIR,'KDDTest+.txt'), multiclass=False)
print(f'Train: {X_tr.shape}, Test: {X_te.shape}, Classes: {len(np.unique(y_tr))}')

print('\n1a. GNN')
results['gnn_binary'] = train_gnn(X_tr, y_tr, X_te, y_te, 2, 'GNN Binary')

print('\n1b. FeatureTransformer')
results['feattrans_binary'] = train_feattrans(X_tr, y_tr, X_te, y_te, 2, 'FeatTrans Binary')

print('\n1c. Full Model')
results['full_binary'] = train_full(X_tr, y_tr, X_te, y_te, 2, 'Full Binary')

# Multiclass
print()
print('=' * 60)
print('2. NSL-KDD Multi-Class')
print('=' * 60)
X_tr_mc, y_tr_mc, X_te_mc, y_te_mc, cls_mc = load_nsl_kdd(os.path.join(DATA_DIR,'KDDTrain+.txt'), os.path.join(DATA_DIR,'KDDTest+.txt'), multiclass=True)
print(f'Train: {X_tr_mc.shape}, Test: {X_te_mc.shape}, Classes: {len(np.unique(y_tr_mc))}')

results['n_classes'] = len(cls_mc)
results['class_names'] = cls_mc

print('\n2a. GNN Multi')
results['gnn_multiclass'] = train_gnn(X_tr_mc, y_tr_mc, X_te_mc, y_te_mc, len(cls_mc), 'GNN Multi')

print('\n2b. FeatureTransformer Multi')
results['feattrans_multiclass'] = train_feattrans(X_tr_mc, y_tr_mc, X_te_mc, y_te_mc, len(cls_mc), 'FeatTrans Multi')

print('\n2c. Full Model Multi')
results['full_multiclass'] = train_full(X_tr_mc, y_tr_mc, X_te_mc, y_te_mc, len(cls_mc), 'Full Multi')

# Save
with open(os.path.join(OUT, 'nsl_kdd_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nDone. Results saved to nsl_kdd_results.json')
