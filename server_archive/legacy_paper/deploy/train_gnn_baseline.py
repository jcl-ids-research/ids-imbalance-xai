"""
Train GNN (GraphSAGE) baseline on UNSW-NB15 for SOTA comparison.
Constructs k-NN graph from tabular features, then applies GraphSAGE.
"""
import os, sys, json, time, argparse, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import kneighbors_graph

# Check if torch_geometric is available
try:
    from torch_geometric.nn import SAGEConv, GCNConv
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print('[WARN] torch_geometric not installed. Will install...')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch_geometric', '-q'])
    from torch_geometric.nn import SAGEConv, GCNConv
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    HAS_PYG = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Data Loading ───
def load_unsw_nb15(data_dir, multiclass=False):
    train_path = os.path.join(data_dir, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(data_dir, 'UNSW_NB15_testing-set.csv')
    
    df_train = pd.read_csv(train_path) if os.path.exists(train_path) else None
    df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None
    
    if df_train is not None and df_test is not None:
        df_all = pd.concat([df_train, df_test], ignore_index=True)
    elif df_train is not None:
        df_all = df_train
    else:
        # Fallback: load partitioned files
        parts = []
        for i in range(1, 5):
            p = os.path.join(data_dir, f'UNSW-NB15_{i}.csv')
            if os.path.exists(p):
                parts.append(pd.read_csv(p))
        df_all = pd.concat(parts, ignore_index=True) if parts else None
    
    if df_all is None:
        raise FileNotFoundError(f'No UNSW-NB15 data found in {data_dir}')
    
    drop_cols = ['id', 'Unnamed: 0']
    df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns], errors='ignore')
    
    feature_cols = [c for c in df_all.columns if c not in ['label', 'attack_cat', 'Label', 'Attack_Cat']]
    X = df_all[feature_cols]
    
    # Encode categoricals
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
    
    if multiclass:
        y_raw = df_all['attack_cat'].astype(str).str.strip().replace('', 'Normal')
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        num_classes = len(le.classes_)
        class_names = list(le.classes_)
        print(f'[DATA] Multiclass: {num_classes} classes: {class_names}')
    else:
        y = df_all['label'].values.astype(np.int64)
        num_classes = 2
        class_names = ['Normal', 'Attack']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=SEED, stratify=y_train)
    
    print(f'[DATA] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
    print(f'[DATA] Class dist train: {np.bincount(y_train)}')
    
    return {
        'X_train': torch.FloatTensor(X_train), 'y_train': torch.LongTensor(y_train),
        'X_val': torch.FloatTensor(X_val), 'y_val': torch.LongTensor(y_val),
        'X_test': torch.FloatTensor(X_test), 'y_test': torch.LongTensor(y_test),
        'input_dim': X_scaled.shape[1], 'num_classes': num_classes,
        'class_names': class_names,
    }

# ─── Graph Construction ───
def build_knn_graph(X, k=10, mode='connectivity'):
    """Build k-NN graph adjacency matrix from feature matrix."""
    print(f'[GRAPH] Building k-NN graph (k={k}, {X.shape[0]} nodes)...')
    adj = kneighbors_graph(X.numpy(), n_neighbors=k, mode=mode, include_self=False)
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    print(f'[GRAPH] Edges: {edge_index.shape[1]}')
    return edge_index

# ─── GraphSAGE Model ───
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

# ─── Training ───
def train_gnn(model, x, edge_index, y_train, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y_train[train_mask])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_gnn(model, x, edge_index, y, mask):
    model.eval()
    out = model(x, edge_index)
    preds = out[mask].argmax(dim=1)
    labels = y[mask]
    acc = accuracy_score(labels.cpu(), preds.cpu())
    p, r, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/opt/UNSW-NB15')
    parser.add_argument('--output_dir', default='/opt/ids_revision/results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--multiclass', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f'Device: {DEVICE}')
    print(f'Config: hidden={args.hidden}, layers={args.layers}, k={args.k}, lr={args.lr}')
    
    # Load data
    data_dict = load_unsw_nb15(args.data_dir, multiclass=args.multiclass)
    
    # Build graph on full dataset for transductive learning
    # Combine train+val+test for graph, use masks
    X_all = torch.cat([data_dict['X_train'], data_dict['X_val'], data_dict['X_test']], dim=0)
    y_all = torch.cat([data_dict['y_train'], data_dict['y_val'], data_dict['y_test']], dim=0)
    
    n_train = len(data_dict['y_train'])
    n_val = len(data_dict['y_val'])
    n_test = len(data_dict['y_test'])
    n_total = n_train + n_val + n_test
    
    print(f'[GRAPH] Total nodes: {n_total}')
    
    # Build k-NN graph
    edge_index = build_knn_graph(X_all, k=args.k)
    
    # Create masks
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True
    
    # Move to device
    X_all = X_all.to(DEVICE)
    y_all = y_all.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)
    
    # Create model
    model = GraphSAGE(
        in_channels=data_dict['input_dim'],
        hidden_channels=args.hidden,
        out_channels=data_dict['num_classes'],
        num_layers=args.layers,
        dropout=0.3,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[MODEL] GraphSAGE: {n_params:,} parameters')
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    best_state = None
    patience = 0
    t0 = time.time()
    
    for epoch in range(1, args.epochs + 1):
        loss = train_gnn(model, X_all, edge_index, y_all, train_mask, optimizer, criterion)
        scheduler.step()
        
        val_acc, val_f1 = evaluate_gnn(model, X_all, edge_index, y_all, val_mask)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            eta = (elapsed / epoch) * (args.epochs - epoch) if epoch > 0 else 0
            print(f'  Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  eta={eta:.0f}s')
        
        if patience >= 15:
            print(f'  Early stopping at epoch {epoch}')
            break
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    # Test
    test_acc, test_f1 = evaluate_gnn(model, X_all, edge_index, y_all, test_mask)
    training_time = time.time() - t0
    
    print(f'\n{"="*60}')
    print(f'[GraphSAGE] Test Results on UNSW-NB15')
    print(f'  Accuracy:  {test_acc:.4f}')
    print(f'  F1 (w):    {test_f1:.4f}')
    print(f'  Params:    {n_params:,}')
    print(f'  Time:      {training_time:.1f}s ({training_time/60:.1f}min)')
    print(f'{"="*60}')
    
    # Save
    result = {
        'model': 'GraphSAGE',
        'dataset': 'UNSW-NB15',
        'n_params': n_params,
        'accuracy': float(test_acc),
        'f1_weighted': float(test_f1),
        'training_time_sec': training_time,
        'epochs_trained': epoch,
        'best_val_f1': float(best_val_f1),
        'config': {
            'hidden_channels': args.hidden,
            'num_layers': args.layers,
            'k_neighbors': args.k,
            'learning_rate': args.lr,
        }
    }
    
    name = 'multiclass' if args.multiclass else 'binary'
    result_path = os.path.join(args.output_dir, f'UNSW-NB15_GraphSAGE_{name}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Results saved to {result_path}')
    
    return result

if __name__ == '__main__':
    main()
