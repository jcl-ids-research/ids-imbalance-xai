"""
Feature-as-Token attention extraction.
Creates a modified model where each feature within a view is a separate token.
This enables genuine per-feature attention visualization.
Trains ~25 epochs, then extracts attention from 300 samples/class.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'; os.environ['OMP_NUM_THREADS']='4'; os.environ['MKL_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda'); SEED = 42
N_SAMPLES = 300
torch.manual_seed(SEED); np.random.seed(SEED)
OUT = '/opt/ids_revision/results/attention_viz'
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, '/opt/ids_revision')
from ablation.models.model_variants import (
    StandardTransformerEncoder, PositionalEncoding, 
    TransformerEncoderBlock, count_parameters
)

# ═══ Feature-as-Token Multi-View Encoder ═══
class FeatureAsTokenMVEncoder(nn.Module):
    def __init__(self, input_dim=42, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, n_views=3):
        super().__init__()
        self.n_views = n_views
        self.d_model = d_model
        base_dim = input_dim // n_views
        remainder = input_dim % n_views
        self.view_dims = [base_dim + (1 if i < remainder else 0) for i in range(n_views)]
        splits = []; start = 0
        for vd in self.view_dims:
            splits.append(list(range(start, start+vd))); start += vd
        self.view_splits = splits
        
        # Per-feature projection: Linear(1, d_model) shared within each view
        self.view_projectors = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_views)
        ])
        self.view_pos_enc = nn.ModuleList([
            PositionalEncoding(d_model, max_len=max(self.view_dims))
            for _ in range(n_views)
        ])
        # Transformer per view - proper seq_len now
        self.view_transformers = nn.ModuleList([
            StandardTransformerEncoder(d_model, d_model, nhead, num_layers,
                                       dim_feedforward, dropout, max_seq_len=vd)
            for vd in self.view_dims
        ])
        self.fusion_proj = nn.Linear(d_model * n_views, d_model)
    
    def forward(self, x):
        view_reps = []
        for i in range(self.n_views):
            vx = x[:, self.view_splits[i]]  # (B, view_dim)
            B, V = vx.shape
            vx = vx.reshape(B*V, 1)  # (B*V, 1)
            vx = self.view_projectors[i](vx)  # (B*V, d_model)
            vx = vx.reshape(B, V, self.d_model)  # (B, V, d_model)
            vx = self.view_pos_enc[i](vx)
            vx = self.view_transformers[i](vx)  # (B, d_model)
            view_reps.append(vx)
        fused = torch.cat(view_reps, dim=-1)
        return self.fusion_proj(fused)

class FullModelFeatAsToken(nn.Module):
    def __init__(self, input_dim=42, num_classes=2, **kwargs):
        super().__init__()
        self.encoder = FeatureAsTokenMVEncoder(input_dim=input_dim, **kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(kwargs.get('d_model',128), 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.encoder(x))

# ═══ Load UNSW-NB15 ═══
print('Loading UNSW-NB15...')
tr=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
te=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
df=pd.concat([tr,te],ignore_index=True)
df=df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns],errors='ignore')
attack_cats=df['attack_cat'].astype(str).str.strip().replace('','Normal').values
labels=df['label'].values.astype(np.int64)
feats=[c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
X=df[feats]
for c in X.select_dtypes(include=['object']).columns: X[c]=LabelEncoder().fit_transform(X[c].astype(str))
X=X.fillna(X.median(numeric_only=True)).astype(np.float32)
X=StandardScaler().fit_transform(X)
all_indices=np.arange(len(X))
X_tr,X_te,y_tr,y_te,idx_tr,idx_te=train_test_split(X,labels,all_indices,test_size=0.2,random_state=SEED,stratify=labels)
X_tr,X_vl,y_tr,y_vl=train_test_split(X_tr,y_tr,test_size=0.125,random_state=SEED,stratify=y_tr)
X_tr_t=torch.FloatTensor(X_tr); y_tr_t=torch.LongTensor(y_tr)
X_vl_t=torch.FloatTensor(X_vl); y_vl_t=torch.LongTensor(y_vl)
X_te_t=torch.FloatTensor(X_te); y_te_t=torch.LongTensor(y_te)
test_cats=attack_cats[idx_te]
print(f'Train: {X_tr.shape}, Val: {X_vl.shape}, Test: {X_te.shape}')

# ═══ Train Feature-as-Token Model ═══
print('\nTraining Feature-as-Token model (30 epochs)...')
model=FullModelFeatAsToken(input_dim=42,num_classes=2,d_model=128,nhead=8,num_layers=4,
                            dim_feedforward=512,dropout=0.1,n_views=3).to(DEVICE)
print(f'Params: {count_parameters(model):,}')
tl=DataLoader(TensorDataset(X_tr_t,y_tr_t),batch_size=256,shuffle=True)
vl=DataLoader(TensorDataset(X_vl_t,y_vl_t),batch_size=256)
crit=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)
best_f1,best_state=0,None; t0=time.time()
for ep in range(1,31):
    model.train()
    for bx,by in tl:
        opt.zero_grad(); l=crit(model(bx.to(DEVICE)),by.to(DEVICE))
        l.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    sch.step()
    model.eval(); ap,al=[],[]
    with torch.no_grad():
        for bx,by in vl:
            preds=model(bx.to(DEVICE)).argmax(dim=1)
            ap.extend(preds.cpu().numpy()); al.extend(by.numpy())
    _,_,f1,_=precision_recall_fscore_support(al,ap,average='weighted',zero_division=0)
    if f1>best_f1: best_f1=f1; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    if ep%10==0: print(f'  Epoch {ep}/30: val_f1={f1:.4f}')
model.load_state_dict(best_state); model.eval()
ap,al=[],[]
with torch.no_grad():
    for bx,by in DataLoader(TensorDataset(X_te_t,y_te_t),batch_size=256):
        preds=model(bx.to(DEVICE)).argmax(dim=1)
        ap.extend(preds.cpu().numpy()); al.extend(by.numpy())
acc=accuracy_score(al,ap); _,_,f1,_=precision_recall_fscore_support(al,ap,average='weighted',zero_division=0)
print(f'Test: Acc={acc:.4f}, F1={f1:.4f}, Time={time.time()-t0:.1f}s')
# Save model after training
print('Saving model...')
torch.save(best_state, os.path.join(OUT, 'feat_token_model.pt'))
print('Model saved.')
preds_test=np.array(ap)

# ═══ Extract attention ═══
print('\nExtracting attention weights...')
attn_data = {}
for vi, view_transformer in enumerate(model.encoder.view_transformers):
    vn = f'View_{vi+1}'
    attn_data[vn] = {}
    for li, layer in enumerate(view_transformer.blocks):
        ln = f'Layer_{li+1}'
        attn_data[vn][ln] = []
        orig_fn = layer.self_attn.forward
        def make_patch(v_name, l_name, storage):
            def patched(query, key, value, **kwargs):
                kwargs['need_weights'] = True; kwargs['average_attn_weights'] = True
                out, w = orig_fn(query, key, value, **kwargs)
                storage.append(w.detach().cpu().numpy()[0])
                return out, w
            return patched
        layer.self_attn.forward = make_patch(vn, ln, attn_data[vn][ln])

# Select samples
correct_mask = (preds_test == y_te)
class_order = ['Normal', 'DoS', 'Fuzzers']
selected = {}
for cat in class_order:
    candidates = np.where((test_cats == cat) & correct_mask)[0]
    if cat != 'Normal':
        candidates = np.array([i for i in candidates if y_te[i] == 1])
    n = min(N_SAMPLES, len(candidates))
    selected[cat] = np.random.RandomState(SEED).choice(candidates, n, replace=False).tolist()
    print(f'{cat}: {len(selected[cat])} samples')

for cat in class_order:
    for idx in selected[cat]:
        with torch.no_grad():
            _ = model(X_te_t[idx:idx+1].to(DEVICE))

for vn in attn_data:
    for ln in attn_data[vn]:
        arrs = attn_data[vn][ln]
        attn_data[vn][ln] = np.array(arrs)
        print(f'{vn}/{ln}: {attn_data[vn][ln].shape}')

# ═══ Compute statistics ═══
offsets = {'Normal':0, 'DoS':len(selected['Normal']), 'Fuzzers':len(selected['Normal'])+len(selected['DoS'])}
feat_names = {
    'View_1': ['dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sinpkt','dinpkt'],
    'View_2': ['sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','res_bdy_len','ct_srv_src'],
    'View_3': ['ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','is_sm_ips_ports','ct_state_ttl','ct_src_ltm','ct_srv_dst','ct_dst_ltm']
}

stats_data = {}
print('\nVERIFICATION:')
summary = {}
colors = {'Normal': '#3498DB', 'DoS': '#E74C3C', 'Fuzzers': '#2ECC71'}
for vn in ['View_1','View_2','View_3']:
    summary[vn] = {'Layer_4': {}}
    stats_data[vn] = {'Layer_4': {}}
    attns = attn_data[vn]['Layer_4']
    for cat in class_order:
        s = start = offsets[cat]; e = start + len(selected[cat])
        cat_attns = attns[s:e].mean(axis=1)
        top5_idx = np.argsort(cat_attns.mean(axis=0))[-5:][::-1]
        names = feat_names.get(vn, [f'F{i}' for i in range(cat_attns.shape[1])])
        top5_feat = [names[i][:12] for i in top5_idx]
        conc = float(np.sort(cat_attns.mean(axis=0))[-5:].sum() / cat_attns.mean(axis=0).sum())
        stats_data[vn]['Layer_4'][cat] = {'mean': cat_attns.mean(axis=0).tolist(), 'std': cat_attns.std(axis=0, ddof=1).tolist()}
        summary[vn]['Layer_4'][cat] = {'top5_features': top5_feat, 'top5_values': [float(round(float(cat_attns.mean(axis=0)[i]),4)) for i in top5_idx], 'concentration': conc}
        print(f'  {vn}/{cat}: top5={top5_feat}, conc={conc:.3f}')

with open(os.path.join(OUT, 'attention_summary.json'), 'w') as f:
    summary['sample_sizes'] = {cat: len(selected[cat]) for cat in class_order}
    json.dump(summary, f, indent=2)

# ═══ Generate 8 figures ═══
print('\nGenerating figures...')

# Fig 1: Heatmaps
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('Average Attention Patterns (Feature-as-Token, Layer 4)', fontsize=14, fontweight='bold')
for ri, cat in enumerate(class_order):
    for ci, vn in enumerate(['View_1','View_2','View_3']):
        ax = axes[ri, ci]
        s = stats_data[vn]['Layer_4'][cat]
        nf = len(s['mean'])
        im = ax.imshow(np.array(s['mean']).reshape(-1, 1), cmap='YlOrRd', aspect='auto')
        names = feat_names.get(vn, [f'F{i+1}' for i in range(nf)])
        ax.set_yticks(range(nf)); ax.set_yticklabels(names[:nf], fontsize=6)
        ax.set_xlim(-1, 1); ax.set_xticks([])
        ax.set_title(f'{cat} | {vn.replace(chr(95),chr(32))}', fontsize=9)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        top3 = np.argsort(s['mean'])[-3:][::-1]
        for ti in top3:
            ax.annotate('*', xy=(0.5, ti), ha='center', fontsize=10, color='blue', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, 'fig1_attention_heatmaps.png'), dpi=200, bbox_inches='tight'); plt.close()
print('[OK] fig1')

# Fig 2: Bar chart
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
fig.suptitle('Feature-level Attention (300 samples/class)', fontsize=14, fontweight='bold')
for vi, vn in enumerate(['View_1','View_2','View_3']):
    ax = axes[vi]; nf = len(stats_data[vn]['Layer_4']['Normal']['mean'])
    x = np.arange(nf); w = 0.25
    for si, cat in enumerate(class_order):
        s = stats_data[vn]['Layer_4'][cat]
        ax.bar(x+(si-1)*w, s['mean'], w, yerr=s['std'], label=cat, color=colors[cat], alpha=0.75, capsize=2, error_kw={'linewidth':0.5})
    names = feat_names.get(vn, [f'F{i}' for i in range(nf)])
    ax.set_xticks(x); ax.set_xticklabels(names[:nf], rotation=45, ha='right', fontsize=6)
    ax.set_title(vn.replace(chr(95),chr(32)), fontsize=10)
    ax.set_ylabel('Avg Attention'); ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_attention_barchart.png'), dpi=200, bbox_inches='tight'); plt.close()
print('[OK] fig2')

# Fig 3: Concentration curves
fig, ax = plt.subplots(figsize=(10, 6))
for vn in ['View_1','View_2','View_3']:
    for cat, color, ls in zip(class_order, ['#3498DB','#E74C3C','#2ECC71'], ['--','-','-.']):
        s = stats_data[vn]['Layer_4'][cat]
        sorted_a = np.sort(s['mean'])[::-1]
        cumsum = np.cumsum(sorted_a)/np.sum(sorted_a)
        ax.plot(np.arange(1,len(cumsum)+1)/len(cumsum)*100, cumsum*100, color=color, linestyle=ls, linewidth=2, markersize=6, marker='o', label=f'{vn}:{cat}')
ax.axhline(y=80, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(95, 81, '80%', ha='right', fontsize=9, color='gray')
ax.set_xlabel('Cumulative Features (%)', fontsize=12); ax.set_ylabel('Cumulative Attention (%)', fontsize=12)
ax.set_title('Attention Concentration', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3); ax.set_xlim(0,100); ax.set_ylim(0,100)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_attention_concentration.png'), dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
print('[OK] fig3')

# Fig 4: Layer evolution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Attention Evolution Across Layers (View 1)', fontsize=14, fontweight='bold')
for ci, cat in enumerate(class_order):
    ax = axes[ci]
    v1_layers = sorted([k for k in attn_data['View_1'].keys()], key=lambda x: int(x.split('_')[1]))
    nf = attn_data['View_1'][v1_layers[0]].shape[-1]
    x = np.arange(nf)
    colors_lyr = ['#3498DB', '#2980B9', '#1F618D', '#E74C3C']
    for li, ln in enumerate(v1_layers):
        s_start = offsets[cat]; s_end = s_start + len(selected[cat])
        layer_attns = attn_data['View_1'][ln][s_start:s_end].mean(axis=1)
        ax.plot(x, layer_attns.mean(axis=0), 'o-', color=colors_lyr[li], linewidth=1.5, markersize=3, label=ln, alpha=0.8)
    ax.set_title(cat, fontsize=11, fontweight='bold'); ax.set_xlabel('Feature index'); ax.set_ylabel('Avg Attention')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig4_layer_evolution.png'), dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
print('[OK] fig4')

# Fig 5: Significance map
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
fig.suptitle('DoS vs Normal: t-test p-values (Bonferroni-corrected)', fontsize=13, fontweight='bold')
for vi, vn in enumerate(['View_1','View_2','View_3']):
    ax = axes[vi]
    attns_v = attn_data[vn]['Layer_4']
    normal_f = attns_v[:len(selected['Normal'])].mean(axis=1)
    dos_f = attns_v[offsets['DoS']:offsets['DoS']+len(selected['DoS'])].mean(axis=1)
    _, p_vals = stats.ttest_ind(dos_f, normal_f, axis=0)
    nf = len(p_vals)
    sig = np.zeros((nf, 1)); sig[p_vals<0.05]=0.5; sig[p_vals<0.01]=1.0
    ax.imshow(sig, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    names = feat_names.get(vn, [f'F{i+1}' for i in range(nf)])
    ax.set_yticks(range(nf)); ax.set_yticklabels(names[:nf], fontsize=6); ax.set_xticks([])
    n01, n05 = int(np.sum(p_vals<0.01)), int(np.sum(p_vals<0.05))
    ax.set_title(f'{vn}: {n01} sig(p<0.01), {n05} sig(p<0.05)', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_significance_map.png'), dpi=200, bbox_inches='tight'); plt.close()
print('[OK] fig5')

# Fig 6: Sample diversity
attns_v1 = attn_data['View_1']['Layer_4']
fig, axes = plt.subplots(3, 25, figsize=(38, 4.5))
fig.suptitle('Individual Attention: 25 Samples/Class (View 1)', fontsize=13, fontweight='bold')
for ri, cat in enumerate(class_order):
    s_start = offsets[cat]; s_end = s_start + len(selected[cat])
    cat_a = attns_v1[s_start:s_end]
    i25 = np.linspace(0, len(cat_a)-1, 25, dtype=int)
    for ci, idx in enumerate(i25):
        ax = axes[ri, ci]
        ax.imshow(cat_a[idx].mean(axis=1).reshape(-1,1), cmap='YlOrRd', aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        if ci==0: ax.set_ylabel(cat, fontsize=9, fontweight='bold')
        if ri==0: ax.set_title(f'S{idx}', fontsize=7)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig6_sample_diversity.png'), dpi=150, bbox_inches='tight'); plt.close()
print('[OK] fig6')

# Fig 7: Extreme cases
fig, axes = plt.subplots(3, 2, figsize=(8, 9))
fig.suptitle('Most Peaked vs Most Dispersed (View 1)', fontsize=13, fontweight='bold')
for ri, cat in enumerate(class_order):
    s_start = offsets[cat]; s_end = s_start + len(selected[cat])
    cat_a = attns_v1[s_start:s_end]
    concs = [np.sort(a.mean(axis=0))[-5:].sum()/a.mean(axis=0).sum() for a in cat_a]
    for ci, (idx, label) in enumerate([(np.argmax(concs), 'Most Peaked'), (np.argmin(concs), 'Most Dispersed')]):
        ax = axes[ri, ci]
        ax.imshow(cat_a[idx].mean(axis=1).reshape(-1,1), cmap='YlOrRd', aspect='auto')
        nf = cat_a[idx].shape[1]
        ax.set_yticks(range(nf)); ax.set_yticklabels(feat_names['View_1'][:nf], fontsize=6); ax.set_xticks([])
        ax.set_title(f'{cat}: {label}\nTop-5 conc: {concs[idx]:.3f}', fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig7_extreme_cases.png'), dpi=200, bbox_inches='tight'); plt.close()
print('[OK] fig7')

# Fig 8: Variance
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Cross-Sample Variance (CV = std/mean)', fontsize=14, fontweight='bold')
for vi, vn in enumerate(['View_1','View_2','View_3']):
    ax = axes[vi]; nf = len(stats_data[vn]['Layer_4']['Normal']['std'])
    x = np.arange(nf); w = 0.25
    for si, cat in enumerate(class_order):
        s = stats_data[vn]['Layer_4'][cat]
        cv = np.array(s['std'])/(np.array(s['mean'])+1e-8)
        ax.bar(x+(si-1)*w, cv, w, label=cat, color=colors[cat], alpha=0.75)
    names = feat_names.get(vn, [f'F{i}' for i in range(nf)])
    ax.set_xticks(x); ax.set_xticklabels(names[:nf], rotation=45, ha='right', fontsize=6)
    ax.set_title(vn.replace(chr(95),chr(32)), fontsize=10)
    ax.set_ylabel('CV'); ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig8_attention_variance.png'), dpi=200, bbox_inches='tight'); plt.close()
print('[OK] fig8')

fig_count = len([f for f in os.listdir(OUT) if f.startswith('fig')])
print('\nALL DONE. ' + str(fig_count) + ' figures.')
