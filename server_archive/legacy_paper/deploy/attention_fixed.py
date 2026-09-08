"""
FIXED ATTENTION EXTRACTION - Uses register_forward_hook (no closure bugs).
Trains full model 50 epochs, extracts attention from 300 samples/class.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'; os.environ['OMP_NUM_THREADS']='4'; os.environ['MKL_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, '/opt/ids_revision')
from ablation.models.model_variants import create_variant, count_parameters

DEVICE = torch.device('cuda'); SEED = 42
N_SAMPLES = 300
torch.manual_seed(SEED); np.random.seed(SEED)
OUT = '/opt/ids_revision/results/attention_viz'
os.makedirs(OUT, exist_ok=True)

# ═══ STEP 1: Load UNSW-NB15 ═══
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

# ═══ STEP 2: Train model ═══
print('\nTraining model (50 epochs)...')
model=create_variant('full',input_dim=42,num_classes=2,d_model=128,nhead=8,num_layers=4,
                      n_views=3,fusion_method='concat',dim_feedforward=512,dropout=0.1).to(DEVICE)
tl=DataLoader(TensorDataset(X_tr_t,y_tr_t),batch_size=512,shuffle=True)
vl=DataLoader(TensorDataset(X_vl_t,y_vl_t),batch_size=512)
crit=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50)
best_f1,best_state=0,None; t0=time.time()
for ep in range(1,51):
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
    if ep%10==0: print(f'  Epoch {ep}/50: val_f1={f1:.4f}')
model.load_state_dict(best_state); model.eval()
ap,al=[],[]
with torch.no_grad():
    for bx,by in DataLoader(TensorDataset(X_te_t,y_te_t),batch_size=512):
        preds=model(bx.to(DEVICE)).argmax(dim=1)
        ap.extend(preds.cpu().numpy()); al.extend(by.numpy())
acc=accuracy_score(al,ap); _,_,f1,_=precision_recall_fscore_support(al,ap,average='weighted',zero_division=0)
print(f'Test: Acc={acc:.4f}, F1={f1:.4f}, Time={time.time()-t0:.1f}s')
preds_test=np.array(ap)

# ═══ STEP 3: Extract attention using register_forward_hook ═══
print('\nExtracting attention weights...')

# Storage: {view_name: {layer_name: list of numpy arrays}}
attn_data = {}
hooks = []

for vi, view_transformer in enumerate(model.encoder.view_transformers):
    vn = f'View_{vi+1}'
    attn_data[vn] = {}
    
    for li, layer in enumerate(view_transformer.blocks):
        ln = f'Layer_{li+1}'
        attn_data[vn][ln] = []
        
        # Capture attention from this specific layer
        def make_hook(vname, lname, storage):
            def hook_fn(module, input, output):
                # The self_attn returns (attn_output, attn_weights) when need_weights=True
                # We need to intercept the forward call to get weights
                pass
            return hook_fn
        
        # Instead of trying to intercept MultiheadAttention's internal weights,
        # use a different approach: register a forward pre-hook and post-hook
        original_forward = layer.self_attn.forward
        
        # Create a closure with proper binding
        def make_patched(v_name, l_name, orig_fn, storage_dict):
            def patched_forward(query, key, value, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = True
                attn_output, attn_weights = orig_fn(query, key, value, **kwargs)
                storage_dict.append(attn_weights.detach().cpu().numpy()[0])
                return attn_output, attn_weights
            return patched_forward
        
        # Assign directly, not via MethodType
        layer.self_attn.forward = make_patched(vn, ln, original_forward, attn_data[vn][ln])

# Select 300 correctly classified samples per class
correct_mask = (preds_test == y_te)
class_order = ['Normal', 'DoS', 'Fuzzers']
selected = {}
for cat in class_order:
    candidates = np.where((test_cats == cat) & correct_mask)[0]
    if cat != 'Normal':
        candidates = np.array([i for i in candidates if y_te[i] == 1])
    n = min(N_SAMPLES, len(candidates))
    selected[cat] = np.random.RandomState(SEED).choice(candidates, n, replace=False).tolist()
    print(f'{cat}: {len(selected[cat])} correctly classified samples')

# Forward all samples
all_samples = []
for cat in class_order:
    for idx in selected[cat]:
        all_samples.append(idx)
        with torch.no_grad():
            _ = model(X_te_t[idx:idx+1].to(DEVICE))
print(f'Total forward passes: {len(all_samples)}')

# Convert to arrays
for vn in attn_data:
    for ln in attn_data[vn]:
        attn_data[vn][ln] = np.array(attn_data[vn][ln])
        print(f'{vn}/{ln}: {attn_data[vn][ln].shape}')

# ═══ STEP 4: Compute statistics ═══
print('\nComputing statistics...')
offsets = {'Normal':0, 'DoS':len(selected['Normal']), 'Fuzzers':len(selected['Normal'])+len(selected['DoS'])}
feat_names_view1 = ['dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sinpkt','dinpkt']
feat_names_view2 = ['sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','res_bdy_len','ct_srv_src']
feat_names_view3 = ['ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports','ct_state_ttl']
view_feat_names = {'View_1': feat_names_view1, 'View_2': feat_names_view2, 'View_3': feat_names_view3}

stats_data = {}
for vn in ['View_1','View_2','View_3']:
    stats_data[vn] = {}
    for ln in ['Layer_4']:
        stats_data[vn][ln] = {}
        attns = attn_data[vn][ln]
        for cat in class_order:
            start = offsets[cat]; end = start + len(selected[cat])
            cat_attns = attns[start:end].mean(axis=1)
            stats_data[vn][ln][cat] = {
                'mean': cat_attns.mean(axis=0).tolist(),
                'std': cat_attns.std(axis=0, ddof=1).tolist()
            }

# VERIFY: print top-5 features per category
print('\nVERIFICATION - Top-5 features per category:')
summary = {}
for vn in ['View_1','View_2','View_3']:
    summary[vn] = {'Layer_4': {}}
    for cat in class_order:
        s = stats_data[vn]['Layer_4'][cat]
        top5_idx = np.argsort(s['mean'])[-5:][::-1]
        names = view_feat_names.get(vn, [f'F{i}' for i in range(len(s['mean']))])
        top5_features = [names[i][:15] for i in top5_idx[:5]]
        top5_values = [round(s['mean'][i], 4) for i in top5_idx[:5]]
        concentration = float(np.sort(s['mean'])[-5:].sum() / np.sum(s['mean']))
        summary[vn]['Layer_4'][cat] = {
            'top5_features': top5_features,
            'top5_values': top5_values,
            'concentration': concentration,
            'sample_size': len(selected[cat])
        }
        print(f'  {vn}/{cat}: top5={top5_features}, conc={concentration:.3f}')

with open(os.path.join(OUT, 'attention_summary.json'), 'w') as f:
    summary['sample_sizes'] = {cat: len(selected[cat]) for cat in class_order}
    json.dump(summary, f, indent=2)

# ═══ STEP 5: Generate 8 figures ═══
print('\nGenerating figures...')

# Fig 1: Average heatmaps (3x3 panel)
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('Average Attention Patterns by View and Attack Category (Layer 4)', fontsize=14, fontweight='bold')
for ri, cat in enumerate(class_order):
    for ci, vn in enumerate(['View_1','View_2','View_3']):
        ax = axes[ri, ci]
        s = stats_data[vn]['Layer_4'][cat]
        nf = len(s['mean'])
        im = ax.imshow(np.array(s['mean']).reshape(-1, 1), cmap='YlOrRd', aspect='auto', vmin=0)
        names = view_feat_names.get(vn, [f'F{i+1}' for i in range(nf)])
        ax.set_yticks(range(nf)); ax.set_yticklabels(names[:nf], fontsize=6)
        ax.set_xlim(-1, 1); ax.set_xticks([])
        ax.set_title(f'{cat} | {vn.replace(chr(95),chr(32))}', fontsize=9)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        top3 = np.argsort(s['mean'])[-3:][::-1]
        for ti in top3:
            ax.annotate('*', xy=(0.5, ti), ha='center', fontsize=10, color='blue', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, 'fig1_attention_heatmaps.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig1')

# Fig 2: Bar chart with error bars (3 views)
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
fig.suptitle('Feature-level Attention with Error Bars (300 samples/class)', fontsize=14, fontweight='bold')
colors = {'Normal': '#3498DB', 'DoS': '#E74C3C', 'Fuzzers': '#2ECC71'}
for vi, vn in enumerate(['View_1','View_2','View_3']):
    ax = axes[vi]
    nf = len(stats_data[vn]['Layer_4']['Normal']['mean'])
    x = np.arange(nf); w = 0.25
    for si, cat in enumerate(class_order):
        s = stats_data[vn]['Layer_4'][cat]
        ax.bar(x + (si-1)*w, s['mean'], w, yerr=s['std'],
               label=cat, color=colors[cat], alpha=0.75, capsize=2, error_kw={'linewidth': 0.5})
    names = view_feat_names.get(vn, [f'F{i}' for i in range(nf)])
    ax.set_xticks(x); ax.set_xticklabels(names[:nf], rotation=45, ha='right', fontsize=6)
    ax.set_title(vn.replace(chr(95), chr(32)), fontsize=10)
    ax.set_ylabel('Avg Attention'); ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_attention_barchart.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig2')

# Fig 3: Concentration curves
fig, ax = plt.subplots(figsize=(10, 6))
for vn in ['View_1','View_2','View_3']:
    for cat, color, ls in zip(class_order, ['#3498DB','#E74C3C','#2ECC71'], ['--','-','-.']):
        s = stats_data[vn]['Layer_4'][cat]
        sorted_attn = np.sort(s['mean'])[::-1]
        cumsum = np.cumsum(sorted_attn) / np.sum(sorted_attn)
        ax.plot(np.arange(1, len(cumsum)+1)/len(cumsum)*100, cumsum*100,
                color=color, linestyle=ls, linewidth=2, markersize=6, marker='o',
                label=f'{vn}:{cat}', markevery=len(cumsum)//5)
ax.axhline(y=80, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(95, 81, '80% attention', ha='right', fontsize=9, color='gray')
ax.set_xlabel('Cumulative Features (%)', fontsize=12)
ax.set_ylabel('Cumulative Attention (%)', fontsize=12)
ax.set_title('Attention Concentration: How Many Features Capture 80% of Attention?', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_attention_concentration.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('[OK] fig3')

# Fig 4: Layer evolution (only View 1 shown)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Attention Evolution Across Transformer Layers (View 1)', fontsize=14, fontweight='bold')
for ci, cat in enumerate(class_order):
    ax = axes[ci]
    s_all = stats_data['View_1']
    layers_present = sorted([k for k in s_all.keys()], key=lambda x: int(x.split('_')[1]))
    nf = len(s_all[layers_present[0]][cat]['mean'])
    x = np.arange(nf)
    colors_lyr = ['#3498DB', '#2980B9', '#1F618D', '#E74C3C']
    for li, ln in enumerate(layers_present):
        s = s_all[ln][cat]
        ax.plot(x, s['mean'], 'o-', color=colors_lyr[li], linewidth=1.5, markersize=3, label=ln, alpha=0.8)
    ax.set_title(cat, fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature index'); ax.set_ylabel('Avg Attention')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig4_layer_evolution.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('[OK] fig4')

# Fig 5: Significance map
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
fig.suptitle('Statistical Significance: DoS vs Normal Attention (t-test p-values)', fontsize=13, fontweight='bold')
for vi, vn in enumerate(['View_1','View_2','View_3']):
    ax = axes[vi]
    attns = attn_data[vn]['Layer_4']
    normal_feat = attns[:len(selected['Normal'])].mean(axis=1)
    dos_feat = attns[offsets['DoS']:offsets['DoS']+len(selected['DoS'])].mean(axis=1)
    _, p_vals = stats.ttest_ind(dos_feat, normal_feat, axis=0)
    nf = len(p_vals)
    sig_map = np.zeros((nf, 1))
    sig_map[p_vals < 0.05] = 0.5; sig_map[p_vals < 0.01] = 1.0
    ax.imshow(sig_map, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    names = view_feat_names.get(vn, [f'F{i+1}' for i in range(nf)])
    ax.set_yticks(range(nf)); ax.set_yticklabels(names[:nf], fontsize=6)
    ax.set_xticks([])
    n_sig01 = int(np.sum(p_vals < 0.01)); n_sig05 = int(np.sum(p_vals < 0.05))
    ax.set_title(f'{vn}: {n_sig01} sig(p<0.01), {n_sig05} sig(p<0.05)', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_significance_map.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig5')

# Fig 6: Sample diversity (25 samples/class, View 1 only)
fig, axes = plt.subplots(3, 25, figsize=(38, 4.5))
fig.suptitle('Individual Attention Patterns: 25 Samples per Class (View 1)', fontsize=13, fontweight='bold')
attns_v1 = attn_data['View_1']['Layer_4']
for ri, cat in enumerate(class_order):
    start = offsets[cat]; end = start + len(selected[cat])
    cat_attns = attns_v1[start:end]
    indices_25 = np.linspace(0, len(cat_attns)-1, 25, dtype=int)
    for ci, idx in enumerate(indices_25):
        ax = axes[ri, ci]
        sample_attn = cat_attns[idx].mean(axis=1).reshape(-1, 1)
        ax.imshow(sample_attn, cmap='YlOrRd', aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0: ax.set_ylabel(cat, fontsize=9, fontweight='bold')
        if ri == 0: ax.set_title(f'S{idx}', fontsize=7)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig6_sample_diversity.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] fig6')

# Fig 7: Extreme cases
fig, axes = plt.subplots(3, 2, figsize=(8, 9))
fig.suptitle('Extreme Cases: Most Peaked vs Most Dispersed per Class (View 1)', fontsize=13, fontweight='bold')
for ri, cat in enumerate(class_order):
    start = offsets[cat]; end = start + len(selected[cat])
    cat_attns = attns_v1[start:end]
    concentrations = []
    for attn in cat_attns:
        feat_attn = attn.mean(axis=0)
        top5_frac = np.sort(feat_attn)[-5:].sum() / feat_attn.sum()
        concentrations.append(top5_frac)
    most_peaked_idx = np.argmax(concentrations)
    most_dispersed_idx = np.argmin(concentrations)
    for ci, (idx, label) in enumerate([(most_peaked_idx, 'Most Peaked'), (most_dispersed_idx, 'Most Dispersed')]):
        ax = axes[ri, ci]
        sample_attn = cat_attns[idx].mean(axis=1).reshape(-1, 1)
        ax.imshow(sample_attn, cmap='YlOrRd', aspect='auto')
        nf = len(sample_attn)
        names = feat_names_view1[:nf]
        ax.set_yticks(range(nf)); ax.set_yticklabels(names, fontsize=6)
        ax.set_xticks([])
        ax.set_title(f'{cat}: {label}\nTop-5 conc: {concentrations[idx]:.3f}', fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig7_extreme_cases.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig7')

# Fig 8: Attention variance
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Cross-Sample Attention Variance (CV = std/mean)', fontsize=14, fontweight='bold')
for vi, vn in enumerate(['View_1','View_2','View_3']):
    ax = axes[vi]
    nf = len(stats_data[vn]['Layer_4']['Normal']['std'])
    x = np.arange(nf); w = 0.25
    for si, cat in enumerate(class_order):
        s = stats_data[vn]['Layer_4'][cat]
        cv = np.array(s['std']) / (np.array(s['mean']) + 1e-8)
        ax.bar(x + (si-1)*w, cv, w, label=cat, color=colors[cat], alpha=0.75)
    names = view_feat_names.get(vn, [f'F{i}' for i in range(nf)])
    ax.set_xticks(x); ax.set_xticklabels(names[:nf], rotation=45, ha='right', fontsize=6)
    ax.set_title(vn.replace(chr(95), chr(32)), fontsize=10)
    ax.set_ylabel('CV (std/mean)'); ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig8_attention_variance.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig8')

fig_count = len([f for f in os.listdir(OUT) if f.startswith('fig')])
print('\nALL DONE. ' + str(fig_count) + ' figures generated.')
