"""
Extract multi-view attention weights from full model for interpretability visualization.
Generates heatmaps for sample flows: Normal, DoS attack, Fuzzers attack.
"""
import os, sys, json, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '4'; os.environ['OMP_NUM_THREADS'] = '4'; os.environ['MKL_NUM_THREADS'] = '4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/opt/ids_revision')
from ablation.models.model_variants import create_variant

DEVICE = torch.device('cuda')
OUT_DIR = '/opt/ids_revision/results/attention_viz'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load UNSW-NB15 ───
def load_data():
    tr = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
    te = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
    df = pd.concat([tr, te], ignore_index=True)
    df = df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns], errors='ignore')
    
    # Save original attack_cat for labeling
    attack_cat = df['attack_cat'].astype(str).str.strip().replace('', 'Normal')
    label = df['label'].values
    
    feats = [c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
    X = df[feats]
    for c in X.select_dtypes(include=['object']).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X = StandardScaler().fit_transform(X)
    
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, label, np.arange(len(X)), test_size=0.2, random_state=42, stratify=label)
    
    return torch.FloatTensor(X_te), np.array(y_te), np.array(attack_cat)[idx_te]

print('Loading data...')
X_test, y_test, cats_test = load_data()
print(f'Test samples: {X_test.shape[0]}')

# ─── Create model and hook ───
print('Creating model...')
model = create_variant('full', input_dim=42, num_classes=2,
                       d_model=128, nhead=8, num_layers=4, n_views=3,
                       fusion_method='concat', dim_feedforward=512, dropout=0.1).to(DEVICE)
model.eval()
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# Hook to capture attention weights from each view's encoder
attention_weights = {}

def hook_view(view_idx):
    def hook_fn(module, input, output):
        pass  # We'll use register_forward_hook on attention layers directly
    return hook_fn

# Register hooks on MultiheadAttention modules
for name, module in model.named_modules():
    if isinstance(module, torch.nn.MultiheadAttention):
        # Store weights during forward pass
        def make_hook(layer_name):
            def hook(module, input, output):
                # input[0] is query, we need to capture attention weights
                # For MultiheadAttention, we can access attn_output_weights
                pass
            return hook

# ─── Select samples ───
# Find one Normal, one DoS, one Fuzzers
sample_indices = {}
for cat in ['Normal', 'DoS', 'Fuzzers']:
    matches = np.where(cats_test == cat)[0]
    if len(matches) > 0:
        sample_indices[cat] = matches[0]
        # Also find attack samples with label=1
        if cat != 'Normal':
            attack_matches = [i for i in matches if y_test[i] == 1]
            if attack_matches:
                sample_indices[cat] = attack_matches[0]

print(f'Selected samples: {sample_indices}')

# ─── Extract attention ───
# We need to forward each sample and capture attention from all layers
# The MultiViewTransformerEncoder has per-view encoders
# Let's modify the model to return attention weights

# Monkey-patch: store attention weights during forward
attention_data = {}  # {view_name: {layer_idx: attention_matrix}}

# The model has: encoder (MultiViewTransformerEncoder) with self.view_transformers = ModuleList
# Each view_transformer is StandardTransformerEncoder with self.blocks = ModuleList of TransformerEncoderBlock
# Each block has self.self_attn (MultiheadAttention)

for view_idx, view_transformer in enumerate(model.encoder.view_transformers):
    view_name = f'View {view_idx+1}'
    attention_data[view_name] = {}
    
    for layer_idx, layer in enumerate(view_transformer.blocks):
        # Monkey-patch this attention layer to capture weights
        original_forward = layer.self_attn.forward
        
        def make_patched(v_name, l_idx, orig):
            def patched_forward(self, query, key, value, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = True
                attn_output, attn_weights = orig(query, key, value, **kwargs)
                attention_data[v_name][f'Layer {l_idx+1}'] = attn_weights.detach().cpu().numpy()
                return attn_output, attn_weights
            return patched_forward
        
        import types
        layer.self_attn.forward = types.MethodType(make_patched(view_name, layer_idx, original_forward), layer.self_attn)
        layer._original_forward = original_forward

# Forward selected samples
results = {}
for cat_name, idx in sample_indices.items():
    x = X_test[idx:idx+1].to(DEVICE)
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()
        results[cat_name] = {
            'prediction': 'Attack' if pred == 1 else 'Normal',
            'true_label': cat_name,
            'f1_score_class': output.softmax(dim=1)[0, pred].item()
        }

# ─── Generate visualizations ───
print('Generating heatmaps...')

# Feature names for each view (simplified labels)
view_features = {
    'View 1': [f'F{i+1}' for i in range(15)],  # 15 features
    'View 2': [f'F{i+1}' for i in range(14)],  # 14 features
    'View 3': [f'F{i+1}' for i in range(13)],  # 13 features
}

# For each selected sample, create a figure with 3 views x last layer attention
for cat_name, idx in sample_indices.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Multi-View Attention — {cat_name} Sample (Predicted: {results[cat_name]["prediction"]})', 
                 fontsize=14, fontweight='bold')
    
    for vi, view_name in enumerate(['View 1', 'View 2', 'View 3']):
        ax = axes[vi]
        data = attention_data.get(view_name, {})
        
        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(view_name)
            continue
        
        # Use last layer attention
        layer_keys = sorted(data.keys())
        last_layer = layer_keys[-1] if layer_keys else None
        
        if last_layer and data[last_layer] is not None:
            attn = data[last_layer]
            # attn shape: (batch, seq_len, seq_len) — typically (1, seq_len, seq_len) with average_attn_weights=True
            if len(attn.shape) == 3:
                attn = attn[0]  # first batch
            
            n_features = attn.shape[0]
            feat_labels = view_features.get(view_name, [f'F{i+1}' for i in range(n_features)])[:n_features]
            
            im = ax.imshow(attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(n_features))
            ax.set_yticks(range(n_features))
            ax.set_xticklabels(feat_labels, fontsize=6, rotation=90)
            ax.set_yticklabels(feat_labels, fontsize=6)
            ax.set_title(f'{view_name} ({n_features} features)\n{last_layer} Attention', fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'No attention data', ha='center', va='center')
            ax.set_title(view_name)
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'attention_{cat_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] {path}')

# Also create a summary figure: average attention across all views for each sample
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Attention Pattern Comparison Across Attack Types', fontsize=14, fontweight='bold')

sample_labels = list(sample_indices.keys())
for si, cat_name in enumerate(sample_labels):
    ax = axes[si]
    
    # Aggregate attention across all views for the last layer
    all_attn = []
    for view_name in ['View 1', 'View 2', 'View 3']:
        data = attention_data.get(view_name, {})
        layer_keys = sorted(data.keys())
        if layer_keys and data[layer_keys[-1]] is not None:
            attn = data[layer_keys[-1]]
            if len(attn.shape) == 3:
                attn = attn[0]
            # Average attention received by each feature
            avg_attn = attn.mean(axis=0)  # shape: (seq_len,)
            all_attn.append(avg_attn)
    
    if all_attn:
        combined = np.concatenate(all_attn)
        ax.bar(range(len(combined)), combined, color='steelblue', alpha=0.7)
        ax.set_title(f'{cat_name}\n(pred: {results[cat_name]["prediction"]})', fontsize=11)
        ax.set_xlabel('Feature index (View1+View2+View3)')
        ax.set_ylabel('Avg Attention')
        ax.set_ylim(0, max(combined) * 1.1)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center')

plt.tight_layout()
path = os.path.join(OUT_DIR, 'attention_summary.png')
fig.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f'[OK] {path}')

print('\nDone. All attention visualizations saved.')
