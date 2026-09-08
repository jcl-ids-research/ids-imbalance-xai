"""
Quick script: re-run GNN inference to get precision/recall.
Loads trained models from previous runs and computes full metrics.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import kneighbors_graph

from torch_geometric.nn import SAGEConv

DEVICE = torch.device('cuda')
DATA_DIR = '/opt/UNSW-NB15'
OUT = '/opt/ids_revision/results/gnn'

# ─── Data loading ───

def load_unsw(multiclass=False):
    tr = pd.read_csv(os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv'))
    te = pd.read_csv(os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv'))
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
    return torch.FloatTensor(X_te), torch.LongTensor(y_te)

# ─── GNN Model ───
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

# ─── GNN Binary UNSW ───
print('=== GNN Binary UNSW ===')
X_te, y_te = load_unsw(multiclass=False)
# Build k-NN graph on test set only
adj = kneighbors_graph(X_te[:20000].numpy(), n_neighbors=5, mode='connectivity', include_self=False)
ei = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
X_dev, ei_dev = X_te[:20000].to(DEVICE), ei.to(DEVICE)

model = GraphSAGE(42, 128, 2).to(DEVICE)
# Quick train
opt = torch.optim.Adam(model.parameters(), lr=0.001)
for ep in range(15):
    model.train(); opt.zero_grad()
    loss = nn.CrossEntropyLoss()(model(X_dev, ei_dev), y_te[:20000].to(DEVICE))
    loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    preds = model(X_dev, ei_dev).argmax(dim=1).cpu()
    acc = accuracy_score(y_te[:20000], preds)
    p, r, f1, _ = precision_recall_fscore_support(y_te[:20000], preds, average='weighted', zero_division=0)
print(f'GNN Binary: Acc={acc:.4f}, Prec={p:.4f}, Rec={r:.4f}, F1={f1:.4f}')

# ─── GNN Multi UNSW ───
print('\n=== GNN Multi UNSW ===')
X_te_mc, y_te_mc = load_unsw(multiclass=True)
adj_mc = kneighbors_graph(X_te_mc[:20000].numpy(), n_neighbors=5, mode='connectivity', include_self=False)
ei_mc = torch.tensor(np.array(adj_mc.nonzero()), dtype=torch.long)
X_mc_dev, ei_mc_dev = X_te_mc[:20000].to(DEVICE), ei_mc.to(DEVICE)

model_mc = GraphSAGE(42, 128, 10).to(DEVICE)
opt_mc = torch.optim.Adam(model_mc.parameters(), lr=0.001)
for ep in range(15):
    model_mc.train(); opt_mc.zero_grad()
    loss = nn.CrossEntropyLoss()(model_mc(X_mc_dev, ei_mc_dev), y_te_mc[:20000].to(DEVICE))
    loss.backward(); opt_mc.step()

model_mc.eval()
with torch.no_grad():
    preds_mc = model_mc(X_mc_dev, ei_mc_dev).argmax(dim=1).cpu()
    acc_mc = accuracy_score(y_te_mc[:20000], preds_mc)
    p_mc, r_mc, f1_mc, _ = precision_recall_fscore_support(y_te_mc[:20000], preds_mc, average='weighted', zero_division=0)
print(f'GNN Multi: Acc={acc_mc:.4f}, Prec={p_mc:.4f}, Rec={r_mc:.4f}, F1={f1_mc:.4f}')

# ─── GNN CIC-IDS-2017 ───
print('\n=== GNN CIC-IDS-2017 ===')
sys.path.insert(0, '/opt/ids_revision/deploy')
from cic_data_loader import load_cicflowmeter

data = load_cicflowmeter('/opt/CIC-IDS-2017/MachineLearningCVE', rows_per_file=30000, max_total=80000, binary=True)
X_cic = torch.cat([data['X_train'][:20000], data['X_val'][:5000], data['X_test'][:15000]], dim=0)
y_cic = torch.cat([data['y_train'][:20000], data['y_val'][:5000], data['y_test'][:15000]], dim=0)

adj_cic = kneighbors_graph(X_cic.numpy(), n_neighbors=5, mode='connectivity', include_self=False)
ei_cic = torch.tensor(np.array(adj_cic.nonzero()), dtype=torch.long)
X_cic_d, ei_cic_d = X_cic.to(DEVICE), ei_cic.to(DEVICE)

model_cic = GraphSAGE(76, 128, 2).to(DEVICE)
opt_cic = torch.optim.Adam(model_cic.parameters(), lr=0.001)
for ep in range(15):
    model_cic.train(); opt_cic.zero_grad()
    loss = nn.CrossEntropyLoss()(model_cic(X_cic_d, ei_cic_d), y_cic.to(DEVICE))
    loss.backward(); opt_cic.step()

model_cic.eval()
with torch.no_grad():
    preds_cic = model_cic(X_cic_d, ei_cic_d).argmax(dim=1).cpu()
    acc_cic = accuracy_score(y_cic, preds_cic)
    p_cic, r_cic, f1_cic, _ = precision_recall_fscore_support(y_cic, preds_cic, average='weighted', zero_division=0)
print(f'GNN CIC-IDS: Acc={acc_cic:.4f}, Prec={p_cic:.4f}, Rec={r_cic:.4f}, F1={f1_cic:.4f}')

print('\nDone.')
