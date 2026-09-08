"""Train GNN (GraphSAGE) on CICFlowMeter datasets (CIC-IDS-2017 / CIC-DDoS2019)."""
import os, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torch_geometric.nn import SAGEConv
warnings.filterwarnings('ignore')

sys.path.insert(0, '/opt/ids_revision/deploy')
from cic_data_loader import load_cicflowmeter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42)

class GraphSAGE(nn.Module):
    def __init__(self, in_c, hid, out_c, layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_c, hid)])
        for _ in range(layers-2):
            self.convs.append(SAGEConv(hid, hid))
        self.convs.append(SAGEConv(hid, hid))
        self.dropout = dropout
        self.cls = nn.Linear(hid, out_c)
    def forward(self, x, ei):
        for c in self.convs:
            x = F.relu(c(x, ei)); x = F.dropout(x, p=self.dropout, training=self.training)
        return self.cls(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/opt/CIC-IDS-2017/MachineLearningCVE')
    parser.add_argument('--output', default='/opt/ids_revision/results')
    parser.add_argument('--dataset', default='CIC-IDS-2017')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=100000)
    args = parser.parse_args()
    
    print(f'Device: {DEVICE}')
    print(f'Loading {args.dataset}...')
    
    data = load_cicflowmeter(args.data_dir, rows_per_file=30000, max_total=args.max_samples, binary=True)
    
    X_all = torch.cat([data['X_train'], data['X_val'], data['X_test']], dim=0)
    y_all = torch.cat([data['y_train'], data['y_val'], data['y_test']], dim=0)
    n_train, n_val, n_test = len(data['y_train']), len(data['y_val']), len(data['y_test'])
    n_total = n_train + n_val + n_test
    
    print(f'Nodes: {n_total}, Features: {data["input_dim"]}, Classes: {data["num_classes"]}')
    
    # Build graph
    print(f'Building k-NN graph (k={args.k})...')
    t0 = time.time()
    adj = kneighbors_graph(X_all.numpy(), n_neighbors=args.k, mode='connectivity', include_self=False)
    ei = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    print(f'Graph built: {ei.shape[1]} edges in {time.time()-t0:.1f}s')
    
    # Masks
    tr_mask = torch.zeros(n_total, dtype=torch.bool); tr_mask[:n_train] = True
    vl_mask = torch.zeros(n_total, dtype=torch.bool); vl_mask[n_train:n_train+n_val] = True
    te_mask = torch.zeros(n_total, dtype=torch.bool); te_mask[n_train+n_val:] = True
    
    X_all, y_all, ei = X_all.to(DEVICE), y_all.to(DEVICE), ei.to(DEVICE)
    tr_mask, vl_mask, te_mask = tr_mask.to(DEVICE), vl_mask.to(DEVICE), te_mask.to(DEVICE)
    
    model = GraphSAGE(data['input_dim'], args.hidden, data['num_classes'], args.layers).to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    
    best_f1, best_st, patience = 0, None, 0
    
    for ep in range(1, args.epochs+1):
        model.train(); opt.zero_grad()
        out = model(X_all, ei)
        loss = crit(out[tr_mask], y_all[tr_mask])
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        
        model.eval()
        with torch.no_grad():
            preds = model(X_all, ei)[vl_mask].argmax(dim=1)
            _, _, f1, _ = precision_recall_fscore_support(y_all[vl_mask].cpu(), preds.cpu(), average='weighted', zero_division=0)
        
        if f1 > best_f1: best_f1 = f1; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else: patience += 1
        
        if ep % 10 == 0 or ep == 1:
            print(f'  Epoch {ep:3d}/{args.epochs}  loss={loss.item():.4f}  val_f1={f1:.4f}')
        if patience >= 15: print(f'  Early stop at {ep}'); break
    
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        preds = model(X_all, ei)[te_mask].argmax(dim=1)
        acc = accuracy_score(y_all[te_mask].cpu(), preds.cpu())
        _, _, f1, _ = precision_recall_fscore_support(y_all[te_mask].cpu(), preds.cpu(), average='weighted', zero_division=0)
    
    print(f'\n[GraphSAGE] {args.dataset}: Acc={acc:.4f}, F1={f1:.4f}, Time={time.time()-t0:.1f}s')
    
    res = {'model': 'GraphSAGE', 'dataset': args.dataset, 'n_params': sum(p.numel() for p in model.parameters()),
           'accuracy': float(acc), 'f1_weighted': float(f1), 'training_time_sec': time.time()-t0}
    path = os.path.join(args.output, f'{args.dataset}_GraphSAGE_binary.json')
    with open(path, 'w') as f: json.dump(res, f, indent=2)
    print(f'Saved: {path}')

if __name__ == '__main__': main()
