"""
Train proper Transformer baseline (feature-as-token) + multi-class eval.
Run on server with GPU.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import sys, json, time, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation.models.feature_transformer import FeatureTransformer, count_parameters

# ── Config ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4
D_MODEL = 64
NHEAD = 8
N_LAYERS = 4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_unsw_nb15(data_dir, multiclass=False):
    """Load UNSW-NB15. If multiclass, use attack_cat as target."""
    train_path = os.path.join(data_dir, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(data_dir, 'UNSW_NB15_testing-set.csv')

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    drop_cols = ['id', 'Unnamed: 0']
    df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns], errors='ignore')

    if multiclass:
        target_col = 'attack_cat'
        y_raw = df_all[target_col].astype(str).str.strip()
        y_raw = y_raw.replace('', 'Normal')
    else:
        target_col = 'label'
        y_raw = df_all[target_col].values

    feature_cols = [c for c in df_all.columns if c not in ['label', 'attack_cat', 'Label', 'Attack_Cat']]
    X = df_all[feature_cols]

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)

    if multiclass:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        num_classes = len(le.classes_)
        print(f'Multiclass: {num_classes} classes: {list(le.classes_)}')
    else:
        y = y_raw.astype(np.int64)
        num_classes = 2

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=SEED, stratify=y_train)

    # SMOTE
    try:
        smote = SMOTE(random_state=SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f'[SMOTE] Train: {X_train.shape}, dist: {np.bincount(y_train)}')
    except:
        pass

    return {
        'X_train': torch.FloatTensor(X_train), 'y_train': torch.LongTensor(y_train),
        'X_val': torch.FloatTensor(X_val), 'y_val': torch.LongTensor(y_val),
        'X_test': torch.FloatTensor(X_test), 'y_test': torch.LongTensor(y_test),
        'input_dim': X_scaled.shape[1], 'num_classes': num_classes,
        'feature_names': feature_cols,
        'class_names': list(le.classes_) if multiclass else ['Normal', 'Attack'],
    }

def load_cic_ddos2019(data_dir):
    """Load CIC-DDoS2019 data."""
    import glob
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    pieces = []
    for f in csv_files[:8]:  # sample from 8 files
        df = pd.read_csv(f, nrows=10000, low_memory=False)
        df.columns = df.columns.str.strip()
        if 'Label' in df.columns:
            pieces.append(df)
    df = pd.concat(pieces, ignore_index=True)

    y_raw = df['Label'].astype(str).str.strip()
    drop_cols = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port',
                 'Destination IP', 'Destination Port', 'Protocol',
                 'Timestamp', 'SimillarHTTP', 'Inbound']
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols + ['Label'], errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    X.replace([np.inf, -np.inf], 0, inplace=True)

    # Binary
    y_bin = np.array([0 if v.upper() == 'BENIGN' else 1 for v in y_raw])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_bin, test_size=0.2, random_state=SEED, stratify=y_bin)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=SEED, stratify=y_train)

    try:
        smote = SMOTE(random_state=SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except:
        pass

    return {
        'X_train': torch.FloatTensor(X_train), 'y_train': torch.LongTensor(y_train),
        'X_val': torch.FloatTensor(X_val), 'y_val': torch.LongTensor(y_val),
        'X_test': torch.FloatTensor(X_test), 'y_test': torch.LongTensor(y_test),
        'input_dim': X_scaled.shape[1], 'num_classes': 2,
    }

def train_and_eval(data, name, output_dir):
    print(f'\n{"="*60}')
    print(f'Training on {name}')
    print(f'Device: {DEVICE}')
    print(f'Features: {data["input_dim"]}, Classes: {data["num_classes"]}')
    print(f'Train: {data["X_train"].shape}, Val: {data["X_val"].shape}, Test: {data["X_test"].shape}')

    model = FeatureTransformer(
        n_features=data['input_dim'],
        d_model=min(D_MODEL, 128),
        nhead=min(NHEAD, 8),
        num_layers=N_LAYERS,
        num_classes=data['num_classes'],
        pooling='mean',
    ).to(DEVICE)
    n_params = count_parameters(model)
    print(f'Model params: {n_params:,}')

    train_loader = DataLoader(TensorDataset(data['X_train'], data['y_train']),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(data['X_val'], data['y_val']),
                            batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(data['X_test'], data['y_test']),
                             batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = 0
    best_state = None
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                preds = model(Xb.to(DEVICE)).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.numpy())
        val_f1 = precision_recall_fscore_support(all_labels, all_preds,
                                                  average='weighted', zero_division=0)[2]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            val_acc = accuracy_score(all_labels, all_preds)
            print(f'  Epoch {epoch:2d}/{EPOCHS}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}')

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds = model(Xb.to(DEVICE)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2]

    print(f'\n  Test:  acc={acc:.4f}  f1={f1:.4f}  f1_macro={f1_macro:.4f}')

    # Per-class metrics
    per_class = {}
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    for cls in labels:
        p, r, f, _ = precision_recall_fscore_support(y_true == cls, y_pred == cls,
                                                      average='binary', zero_division=0)
        per_class[int(cls)] = {'precision': p, 'recall': r, 'f1': f}

    training_time = time.time() - start_time

    # Save
    result = {
        'model': 'FeatureTransformer',
        'dataset': name,
        'n_params': n_params,
        'accuracy': acc,
        'f1_weighted': f1,
        'f1_macro': f1_macro,
        'precision': prec,
        'recall': rec,
        'training_time_sec': training_time,
        'num_classes': data['num_classes'],
        'per_class': {str(k): v for k, v in per_class.items()},
    }

    if 'class_names' in data:
        result['class_names'] = data['class_names']

    os.makedirs(output_dir, exist_ok=True)
    safe_name = name.replace(' ', '_').replace('/', '_')
    path = os.path.join(output_dir, f'{safe_name}_transformer_baseline.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Results saved to {path}')

    # Print classification report for multi-class
    if data['num_classes'] > 2 and 'class_names' in data:
        print('\n  Classification Report:')
        print(classification_report(y_true, y_pred,
              target_names=data['class_names'], zero_division=0, digits=4))

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unsw_dir', default='/opt/UNSW-NB15')
    parser.add_argument('--cic_dir', default='/opt/CIC-DDoS2019/all')
    parser.add_argument('--output_dir', default='/opt/ids_revision/results')
    parser.add_argument('--binary', action='store_true', default=True)
    parser.add_argument('--multiclass', action='store_true')
    parser.add_argument('--no_cic', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # 1. UNSW-NB15 binary
    if args.binary:
        data = load_unsw_nb15(args.unsw_dir, multiclass=False)
        all_results['UNSW-NB15'] = train_and_eval(data, 'UNSW-NB15', args.output_dir)

    # 2. UNSW-NB15 multi-class (for per-class Analysis/Fuzzers)
    if args.multiclass:
        data_mc = load_unsw_nb15(args.unsw_dir, multiclass=True)
        all_results['UNSW-NB15 (multi)'] = train_and_eval(data_mc, 'UNSW-NB15_multiclass', args.output_dir)

    # 3. CIC-DDoS2019
    if not args.no_cic:
        try:
            data_cic = load_cic_ddos2019(args.cic_dir)
            all_results['CIC-DDoS2019'] = train_and_eval(data_cic, 'CIC-DDoS2019', args.output_dir)
        except Exception as e:
            print(f'[SKIP] CIC-DDoS2019: {e}')

    print(f'\n{"="*60}')
    print('Summary:')
    for name, res in all_results.items():
        print(f'  {name:25s}  acc={res["accuracy"]:.4f}  f1={res["f1_weighted"]:.4f}  params={res["n_params"]:,}')
