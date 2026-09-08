"""
Ablation Study: Training and Evaluation Script
==============================================
Trains all 4 model variants on UNSW-NB15 (or your dataset) with:
  - Unified dataloader
  - Consistent hyperparameters
  - Per-class and overall metrics
  - Confusion matrix export
  - CSV result logging

Usage:
    python train_ablation.py --variant full --epochs 50 --data_dir ./data
    python train_ablation.py --all  # train all 4 variants sequentially

Author: [Your Name]
"""

import argparse
import os
import sys
import json
import time
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation.models.model_variants import create_variant, count_parameters

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'n_views': 3,
    'fusion_method': 'concat',
    'dim_feedforward': 512,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 64,
    'epochs': 50,
    'patience': 10,       # early stopping
    'num_classes': 2,     # binary: normal vs attack
    'seed': 42,
}

# ============================================================================
# Data Loading (UNSW-NB15)
# ============================================================================

def load_unsw_nb15(data_dir: str, test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42, use_smote: bool = True,
                   multiclass: bool = False) -> Dict:
    """
    Load UNSW-NB15 dataset from CSV files.
    If multiclass=True, uses 'attack_cat' column as target.
    """
    train_path = os.path.join(data_dir, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(data_dir, 'UNSW_NB15_testing-set.csv')

    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
    else:
        parts = []
        for i in range(1, 5):
            p = os.path.join(data_dir, f'UNSW-NB15_{i}.csv')
            if os.path.exists(p):
                parts.append(pd.read_csv(p))
        if parts:
            df_train = pd.concat(parts, ignore_index=True)
        else:
            print("[WARN] UNSW-NB15 CSV not found. Generating synthetic data for pipeline test.")
            return _generate_synthetic_data(train_size=10000, test_size=2000,
                                            val_size=1000, random_state=random_state)

    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        df_all = pd.concat([df_train, df_test], ignore_index=True)
    else:
        df_all = df_train

    label_col = 'label'
    attack_cat_col = 'attack_cat'

    drop_cols = ['id', 'Unnamed: 0']
    df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns], errors='ignore')

    if label_col not in df_all.columns and 'Label' in df_all.columns:
        label_col = 'Label'
    if attack_cat_col not in df_all.columns and 'Attack_Cat' in df_all.columns:
        attack_cat_col = 'Attack_Cat'

    if multiclass:
        target_col = attack_cat_col
        y_raw = df_all[target_col].astype(str).str.strip()
        y_raw = y_raw.replace('', 'Normal')
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        num_classes = len(le.classes_)
        class_names = list(le.classes_)
        print(f"[MULTICLASS] {num_classes} classes: {class_names}")
    else:
        target_col = label_col
        y = df_all[target_col].values if target_col in df_all.columns else None
        num_classes = len(np.unique(y)) if y is not None else 2
        class_names = ['Normal', 'Attack']

    feature_cols = [c for c in df_all.columns
                    if c not in [label_col, attack_cat_col, 'Label', 'Attack_Cat']]

    X = df_all[feature_cols]

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    X = X.fillna(X.median(numeric_only=True))
    X = X.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if y is None:
        y = np.random.randint(0, 2, size=len(X))

    y = y.astype(np.int64)

    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adjusted, random_state=random_state, stratify=y_temp
    )

    print(f"[DATA] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"[DATA] Class distribution - Train: {np.bincount(y_train)}")

    result = {
        'X_train': torch.FloatTensor(X_train),
        'y_train': torch.LongTensor(y_train),
        'X_val': torch.FloatTensor(X_val),
        'y_val': torch.LongTensor(y_val),
        'X_test': torch.FloatTensor(X_test),
        'y_test': torch.LongTensor(y_test),
        'scaler': scaler,
        'feature_names': feature_cols,
        'input_dim': X_scaled.shape[1],
        'num_classes': len(np.unique(y)),
    }

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            result['X_train'] = torch.FloatTensor(X_train_sm)
            result['y_train'] = torch.LongTensor(y_train_sm)
            print(f"[SMOTE] After SMOTE: {X_train_sm.shape}, "
                  f"distribution: {np.bincount(y_train_sm)}")
        except ImportError:
            print("[WARN] imblearn not installed. Skipping SMOTE.")

    return result


def _generate_synthetic_data(train_size: int, test_size: int, val_size: int,
                             random_state: int, input_dim: int = 49,
                             num_classes: int = 2) -> Dict:
    """Generate synthetic data for pipeline testing."""
    rng = np.random.RandomState(random_state)
    n_total = train_size + test_size + val_size
    X = rng.randn(n_total, input_dim).astype(np.float32)
    y = rng.randint(0, num_classes, size=n_total).astype(np.int64)

    X_train = torch.FloatTensor(X[:train_size])
    y_train = torch.LongTensor(y[:train_size])
    X_val = torch.FloatTensor(X[train_size:train_size+val_size])
    y_val = torch.LongTensor(y[train_size:train_size+val_size])
    X_test = torch.FloatTensor(X[train_size+val_size:])
    y_test = torch.LongTensor(y[train_size+val_size:])

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'feature_names': [f'feature_{i}' for i in range(input_dim)],
    }


# ============================================================================
# Diffusion Augmentation Wrapper
# ============================================================================

class DiffusionAugmentedDataset(Dataset):
    """
    Wraps a base dataset and applies diffusion-based augmentation on-the-fly.
    The diffusion model generates additional synthetic samples per batch.
    """
    def __init__(self, base_dataset: Dataset, diffusion_model: nn.Module,
                 augment_ratio: float = 0.5, device: torch.device = torch.device('cpu')):
        self.base = base_dataset
        self.diffusion = diffusion_model.to(device)
        self.augment_ratio = augment_ratio
        self.device = device

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if torch.rand(1).item() < self.augment_ratio:
            with torch.no_grad():
                x_syn = self.diffusion.sample(1, self.device)
                x = x_syn.squeeze(0).cpu()
        return x, y


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model: nn.Module, dataloader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device, diffusion_model: Optional[nn.Module] = None,
                augment_ratio: float = 0.0) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Optional diffusion augmentation within batch
        if diffusion_model is not None and augment_ratio > 0 and torch.rand(1).item() < augment_ratio:
            n_aug = max(1, int(batch_x.size(0) * 0.2))
            with torch.no_grad():
                syn_x = diffusion_model.sample(n_aug, device)
                syn_y = batch_y[:n_aug]  # assign nearest class (simplified)
            batch_x = torch.cat([batch_x, syn_x], dim=0)
            batch_y = torch.cat([batch_y, syn_y], dim=0)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: torch.device,
             num_classes: int = 2) -> Dict:
    """Evaluate model. Returns dict of metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        total_loss += loss.item()
        num_batches += 1

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Per-class metrics
    per_class = {}
    labels = np.unique(np.concatenate([y_true, y_pred]))
    for cls in labels:
        p, r, f, _ = precision_recall_fscore_support(
            y_true == cls, y_pred == cls, average='binary', zero_division=0
        )
        per_class[int(cls)] = {'precision': p, 'recall': r, 'f1': f}

    cm = confusion_matrix(y_true, y_pred)

    return {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
    }


# ============================================================================
# Main Training Pipeline
# ============================================================================

def run_ablation_experiment(
    variant_name: str,
    data_dir: str = './data',
    output_dir: str = './results',
    config: Optional[Dict] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    verbose: bool = True,
) -> Dict:
    """Run ablation experiment for a single variant."""

    if config is None:
        config = DEFAULT_CONFIG.copy()

    os.makedirs(output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training variant: {variant_name}")
        print(f"Device: {device}")
        print(f"{'='*60}")

    # Load data
    multiclass = config.get('multiclass', False)
    data = load_unsw_nb15(data_dir, random_state=config['seed'], multiclass=multiclass)
    input_dim = data['input_dim']
    num_classes = data.get('num_classes', config['num_classes'])

    # Create model
    model = create_variant(
        variant_name,
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        n_views=config['n_views'],
        fusion_method=config['fusion_method'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
    )
    model = model.to(device)
    n_params = count_parameters(model)

    if verbose:
        print(f"Input dim: {input_dim}, Classes: {num_classes}, Params: {n_params:,}")

    # Dataloaders
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    val_dataset = TensorDataset(data['X_val'], data['y_val'])
    test_dataset = TensorDataset(data['X_test'], data['y_test'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=0)

    # Optimizer & criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Training loop
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_weighted'])

        # Early stopping based on val F1
        if val_metrics['f1_weighted'] > best_val_f1:
            best_val_f1 = val_metrics['f1_weighted']
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{config['epochs']}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}, "
                  f"val_f1={val_metrics['f1_weighted']:.4f}, "
                  f"LR={scheduler.get_last_lr()[0]:.2e}")

        if patience_counter >= config['patience']:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    training_time = time.time() - start_time

    # Compile results
    results = {
        'variant': variant_name,
        'config': config,
        'n_params': n_params,
        'training_time_sec': training_time,
        'training_time_min': training_time / 60,
        'best_val_f1': best_val_f1,
        'test': test_metrics,
        'history': history,
    }

    # Save results
    result_path = os.path.join(output_dir, f'{variant_name}_results.json')
    # Convert non-serializable items
    save_results = {
        'variant': variant_name,
        'n_params': n_params,
        'num_classes': num_classes,
        'training_time_sec': training_time,
        'best_val_f1': best_val_f1,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_weighted': test_metrics['f1_weighted'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_precision': test_metrics['precision_weighted'],
        'test_recall': test_metrics['recall_weighted'],
        'per_class': {str(k): v for k, v in test_metrics['per_class'].items()},
        'confusion_matrix': test_metrics['confusion_matrix'],
        'class_names': data.get('class_names', []),
    }
    with open(result_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    if verbose:
        print(f"\n[{variant_name}] Results saved to {result_path}")
        print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Test F1 (w):    {test_metrics['f1_weighted']:.4f}")
        print(f"  Test F1 (macro):{test_metrics['f1_macro']:.4f}")
        if num_classes > 2 and data.get('class_names'):
            print(classification_report(test_metrics['y_true'], test_metrics['y_pred'],
                  target_names=data['class_names'], zero_division=0, digits=4))

    return results


def run_all_ablation(data_dir: str = './data', output_dir: str = './results',
                     config: Optional[Dict] = None):
    """Run all 4 ablation variants sequentially and produce summary table."""
    variants = ['full', 'wo_diffusion', 'wo_multiview', 'wo_both']
    all_results = {}

    for variant in variants:
        results = run_ablation_experiment(variant, data_dir, output_dir, config)
        all_results[variant] = results

    # Create summary table
    summary_path = os.path.join(output_dir, 'ablation_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant', 'Params', 'Test Acc', 'Test F1 (w)',
                         'Test F1 (macro)', 'Test Precision', 'Test Recall',
                         'Best Val F1', 'Time (min)'])
        for v in variants:
            r = all_results[v]
            writer.writerow([
                v, r['n_params'],
                f"{r['test']['accuracy']:.4f}",
                f"{r['test']['f1_weighted']:.4f}",
                f"{r['test']['f1_macro']:.4f}",
                f"{r['test']['precision_weighted']:.4f}",
                f"{r['test']['recall_weighted']:.4f}",
                f"{r['best_val_f1']:.4f}",
                f"{r['training_time_min']:.2f}",
            ])

    print(f"\n{'='*60}")
    print("Ablation Summary Table")
    print(f"{'='*60}")
    print(f"{'Variant':20s} {'Params':>10s} {'Acc':>8s} {'F1(w)':>8s} {'F1(m)':>8s}")
    print("-" * 60)
    for v in variants:
        r = all_results[v]
        print(f"{v:20s} {r['n_params']:>10,d} "
              f"{r['test']['accuracy']:>8.4f} {r['test']['f1_weighted']:>8.4f} "
              f"{r['test']['f1_macro']:>8.4f}")
    print(f"{'='*60}")
    print(f"Summary saved to {summary_path}")


# ============================================================================
# Command Line Entry
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study for IDS Model')
    parser.add_argument('--variant', type=str, default='full',
                        choices=['full', 'wo_diffusion', 'wo_multiview', 'wo_both', 'all'],
                        help='Model variant to train (default: full)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--multiclass', action='store_true',
                        help='Use attack_cat for multi-class classification')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    config = DEFAULT_CONFIG.copy()
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'multiclass': args.multiclass,
    })

    if args.variant == 'all':
        run_all_ablation(args.data_dir, args.output_dir, config)
    else:
        run_ablation_experiment(args.variant, args.data_dir, args.output_dir, config, device)
