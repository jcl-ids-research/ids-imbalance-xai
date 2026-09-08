"""
CICFlowMeter Data Loader for our Diffusion + Multi-View Transformer.

Loads CIC-IDS-2017 or CIC-DDoS2019 CSV files and returns data in the
same format as load_unsw_nb15() for compatibility with train_ablation.py.
"""
import os, sys, glob, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, Optional, List

CIC_DROP_COLS = [
    'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port',
    'Destination IP', 'Destination Port', 'Protocol',
    'Timestamp', 'SimillarHTTP', 'Inbound', 'Fwd Header Length.1',
]

def load_cicflowmeter(
    data_dir: str,
    rows_per_file: int = 50000,
    max_total: int = 400000,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    binary: bool = True,
    label_col: str = 'Label',
) -> Dict:
    """
    Load CICFlowMeter CSV files (CIC-IDS-2017 or CIC-DDoS2019).
    
    Returns dict compatible with train_ablation.py's data format:
      X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names, input_dim, num_classes
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f'[CIC-LOADER] Found {len(csv_files)} CSV files in {data_dir}')
    
    pieces = []
    total_rows = 0
    for f in csv_files:
        fname = os.path.basename(f)
        try:
            df = pd.read_csv(f, nrows=200000, low_memory=False)
        except Exception as e:
            print(f'  [SKIP] {fname}: read error: {e}')
            continue
        
        df.columns = df.columns.str.strip()
        
        if label_col not in df.columns:
            # Try last column as label
            label_col = df.columns[-1]
            print(f'  [WARN] Using last column as label: "{label_col}"')
        
        n = min(len(df), rows_per_file)
        sampled = df.sample(n=n, random_state=random_state)
        pieces.append(sampled)
        total_rows += len(sampled)
        
        dist = sampled[label_col].value_counts()
        print(f'  {fname}: {len(sampled):,} rows, labels={dict(dist.head(5))}')
        
        if total_rows >= max_total:
            break
    
    df = pd.concat(pieces, ignore_index=True)
    print(f'\n[CIC-LOADER] Total: {len(df):,} rows, {df[label_col].nunique()} classes')
    
    # ---- Preprocessing ----
    # Drop non-feature columns
    drop_cols = [c for c in CIC_DROP_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    # Separate label
    y_raw = df[label_col].astype(str).str.strip()
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols]
    
    # Convert all to numeric, coerce errors
    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Encode labels
    if binary:
        # Map to binary: BENIGN / normal vs Attack
        y_bin = y_raw.map(lambda v: 'BENIGN' if v.upper() in ['BENIGN', 'NORMAL', ''] else 'ATTACK')
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_bin)
        print(f'  Binary mapping: {dict(pd.Series(y_bin).value_counts())}')
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)
        print(f'  Multi-class: {len(le.classes_)} classes')
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    n_features = X_scaled.shape[1]
    n_classes = len(np.unique(y_encoded))
    print(f'  Features: {n_features}, Classes: {n_classes}')
    
    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size,
        random_state=random_state, stratify=y_encoded
    )
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adjusted,
        random_state=random_state, stratify=y_temp
    )
    
    print(f'  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
    print(f'  Train class dist: {np.bincount(y_train)}')
    
    return {
        'X_train': torch.FloatTensor(X_train),
        'y_train': torch.LongTensor(y_train),
        'X_val': torch.FloatTensor(X_val),
        'y_val': torch.LongTensor(y_val),
        'X_test': torch.FloatTensor(X_test),
        'y_test': torch.LongTensor(y_test),
        'scaler': scaler,
        'feature_names': feature_cols,
        'input_dim': n_features,
        'num_classes': n_classes,
    }


if __name__ == '__main__':
    # Quick test
    data = load_cicflowmeter(
        '/opt/CIC-IDS-2017/MachineLearningCVE',
        rows_per_file=10000,
        max_total=80000,
    )
    print(f'\nTest OK. Input dim: {data["input_dim"]}, Classes: {data["num_classes"]}')
    print(f'X_train shape: {data["X_train"].shape}')
