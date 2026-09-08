"""
Train Diffusion+MV-Transformer (full model) on CIC-DDoS2019.
Binary: BENIGN vs Attack (all DDoS types merged).
"""
import os, sys, json, time
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '/opt/ids_revision')
from ablation.models.model_variants import create_variant, count_parameters

sys.path.insert(0, '/opt/ids_revision/deploy')
from cic_data_loader import load_cicflowmeter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)

    print('Loading CIC-DDoS2019 data...', flush=True)
    data = load_cicflowmeter(
        '/opt/CIC-DDoS2019/all',
        rows_per_file=50000, max_total=400000, binary=True,
    )
    input_dim = data['input_dim']
    num_classes = data['num_classes']
    print(f'Input dim: {input_dim}, Classes: {num_classes}', flush=True)
    print(f'Train: {data["X_train"].shape}, Val: {data["X_val"].shape}, Test: {data["X_test"].shape}', flush=True)

    model = create_variant(
        'full', input_dim=input_dim, num_classes=num_classes,
        d_model=128, nhead=8, num_layers=4, n_views=3,
        fusion_method='concat', dim_feedforward=512, dropout=0.1,
    ).to(device)
    print(f'Model params: {count_parameters(model):,}', flush=True)

    B = 128
    tl = DataLoader(TensorDataset(data['X_train'], data['y_train']),
                    batch_size=B, shuffle=True, num_workers=0)
    vl = DataLoader(TensorDataset(data['X_val'], data['y_val']),
                    batch_size=B, shuffle=False, num_workers=0)
    ttl = DataLoader(TensorDataset(data['X_test'], data['y_test']),
                     batch_size=B, shuffle=False, num_workers=0)

    train_steps = len(tl)
    print(f'Train batches: {train_steps}, Val batches: {len(vl)}', flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_f1 = 0.0
    best_state = None
    patience = 0
    max_epochs = 50
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for bx, by in tl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        model.eval()
        ap, al = [], []
        with torch.no_grad():
            for bx, by in vl:
                out = model(bx.to(device))
                ap.extend(torch.argmax(out, dim=1).cpu().numpy())
                al.extend(by.numpy())

        val_f1 = precision_recall_fscore_support(al, ap, average='weighted', zero_division=0)[2]
        val_acc = accuracy_score(al, ap)
        avg_loss = running_loss / train_steps

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        elapsed = time.time() - t0
        eta = (elapsed / epoch) * (max_epochs - epoch) if epoch > 0 else 0

        print(f'Epoch {epoch:2d}/{max_epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  patience={patience}  eta={eta:.0f}s', flush=True)

        if patience >= 10:
            print(f'Early stopping at epoch {epoch}', flush=True)
            break

    print(f'Training time: {time.time()-t0:.1f}s', flush=True)

    model.load_state_dict(best_state)
    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for bx, by in ttl:
            out = model(bx.to(device))
            ap.extend(torch.argmax(out, dim=1).cpu().numpy())
            al.extend(by.numpy())

    y_true, y_pred = np.array(al), np.array(ap)
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print(f'\n========== Final Test Results on CIC-DDoS2019 ==========', flush=True)
    print(f'Accuracy:   {acc:.4f} ({acc*100:.2f}%)', flush=True)
    print(f'F1 (w):     {f:.4f} ({f*100:.2f}%)', flush=True)
    print(f'Precision:  {p:.4f}', flush=True)
    print(f'Recall:     {r:.4f}', flush=True)
    print(f'Total time: {time.time()-t0:.1f}s', flush=True)
    print(f'========================================================', flush=True)

    results = {
        'variant': 'full', 'dataset': 'CIC-DDoS2019',
        'accuracy': float(acc), 'f1_weighted': float(f),
        'precision': float(p), 'recall': float(r),
        'params': count_parameters(model),
        'time_sec': time.time() - t0,
        'epochs_trained': epoch,
        'best_val_f1': float(best_val_f1),
    }

    out_dir = '/opt/ids_revision/results/cic_ddos2019_full'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'full_results.json'), 'w') as fp:
        json.dump(results, fp, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, 'full_model.pt'))
    print(f'Results saved to {out_dir}/', flush=True)

if __name__ == '__main__':
    main()
