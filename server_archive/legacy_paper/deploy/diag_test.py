import sys, os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

sys.path.insert(0, '/opt/ids_revision')
os.chdir('/opt/ids_revision')

print('--- Test 1: import model ---')
from ablation.models.feature_transformer import FeatureTransformer
import torch
m = FeatureTransformer(42, num_classes=2, d_model=128, nhead=8, num_layers=4)
x = torch.randn(2, 42)
out = m(x)
print(f'model output shape: {out.shape}')

print('\n--- Test 2: load UNSW data ---')
from ablation.train_transformer_baseline import load_unsw_nb15
data = load_unsw_nb15('/opt/UNSW-NB15', multiclass=False)
print(f'X_train: {data["X_train"].shape}, X_test: {data["X_test"].shape}')
print(f'input_dim: {data["input_dim"]}, num_classes: {data["num_classes"]}')
print(f'train classes: {set(data["y_train"].tolist())}')

print('\n--- Test 3: run 2 epochs of training ---')
import torch
from ablation.models.feature_transformer import FeatureTransformer
model = FeatureTransformer(data['input_dim'], num_classes=data['num_classes'], d_model=64, nhead=8, num_layers=4)

# Do a quick training loop
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda')
model = model.to(device)
train_ds = TensorDataset(data['X_train'].to(device), data['y_train'].to(device))
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}')

# Quick eval
model.eval()
with torch.no_grad():
    X_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)
    out = model(X_test)
    preds = out.argmax(dim=1)
    acc = (preds == y_test).float().mean().item()
print(f'Test accuracy after 2 epochs: {acc:.4f}')
