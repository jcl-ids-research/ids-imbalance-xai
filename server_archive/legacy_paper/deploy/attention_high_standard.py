"""
HIGH-STANDARD Attention Analysis for UNSW-NB15.
- Train full model 50 epochs, save weights
- 50 correctly classified samples per class (Normal/DoS/Fuzzers)
- Extract attention from ALL 4 layers, ALL 3 views
- Per-feature mean±std, statistical comparison, entropy analysis
- Publication-quality visualizations
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, types, warnings
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

sys.path.insert(0, '/opt/ids_revision')
from ablation.models.model_variants import create_variant, count_parameters

DEVICE = torch.device('cuda')
SEED = 42; N_SAMPLES = 300
torch.manual_seed(SEED); np.random.seed(SEED)

OUT = '/opt/ids_revision/results/attention_viz'
os.makedirs(OUT, exist_ok=True)

# ═══════ STEP 1: Load Data ═══════
print('='*60); print('STEP 1: Loading UNSW-NB15'); print('='*60)
tr = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
te = pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
df = pd.concat([tr, te], ignore_index=True)
df = df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns], errors='ignore')

attack_cats_all = df['attack_cat'].astype(str).str.strip().replace('','Normal').values
labels_all = df['label'].values.astype(np.int64)

feats = [c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
X_all = df[feats]
for c in X_all.select_dtypes(include=['object']).columns:
    X_all[c] = LabelEncoder().fit_transform(X_all[c].astype(str))
X_all = X_all.fillna(X_all.median(numeric_only=True)).astype(np.float32)
X_all = StandardScaler().fit_transform(X_all)

all_indices = np.arange(len(X_all))
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(X_all, labels_all, all_indices, test_size=0.2, random_state=SEED, stratify=labels_all)
X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.125, random_state=SEED, stratify=y_tr)

X_tr_t=torch.FloatTensor(X_tr); y_tr_t=torch.LongTensor(y_tr)
X_vl_t=torch.FloatTensor(X_vl); y_vl_t=torch.LongTensor(y_vl)
X_te_t=torch.FloatTensor(X_te); y_te_t=torch.LongTensor(y_te)

test_cats = attack_cats_all[idx_te]
print(f'Train: {X_tr.shape}, Val: {X_vl.shape}, Test: {X_te.shape}')

# ═══════ STEP 2: Train Full Model ═══════
print(); print('='*60); print('STEP 2: Training Full Model (50 epochs)'); print('='*60)
model = create_variant('full', input_dim=42, num_classes=2, d_model=128, nhead=8, num_layers=4,
                       n_views=3, fusion_method='concat', dim_feedforward=512, dropout=0.1).to(DEVICE)
print(f'Params: {count_parameters(model):,}')

tl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True)
vl = DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=512)

crit=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

best_f1, best_state = 0, None; t0=time.time()
for ep in range(1, 51):
    model.train()
    for bx,by in tl:
        bx,by=bx.to(DEVICE),by.to(DEVICE); opt.zero_grad()
        loss=crit(model(bx),by); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    sch.step()
    model.eval(); ap,al=[],[]
    with torch.no_grad():
        for bx,by in vl:
            preds=model(bx.to(DEVICE)).argmax(dim=1)
            ap.extend(preds.cpu().numpy()); al.extend(by.numpy())
    _,_,f1,_=precision_recall_fscore_support(al,ap,average='weighted',zero_division=0)
    if f1>best_f1: best_f1=f1; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    if ep%10==0: print(f'  Epoch {ep:2d}/50  val_acc={accuracy_score(al,ap):.4f}  val_f1={f1:.4f}')

model.load_state_dict(best_state)
model.eval(); ap,al=[],[]
with torch.no_grad():
    for bx,by in DataLoader(TensorDataset(X_te_t,y_te_t),batch_size=512):
        preds=model(bx.to(DEVICE)).argmax(dim=1)
        ap.extend(preds.cpu().numpy()); al.extend(by.numpy())
acc=accuracy_score(al,ap); _,_,f1,_=precision_recall_fscore_support(al,ap,average='weighted',zero_division=0)
print(f'Test: Acc={acc:.4f}, F1={f1:.4f}, Time={time.time()-t0:.1f}s')
preds_test = np.array(ap)

torch.save(best_state, os.path.join(OUT,'full_model_trained.pt'))

# ═══════ STEP 3: Select Samples & Extract Attention ═══════
print(); print('='*60); print('STEP 3: Selecting samples & extracting attention'); print('='*60)

# Find correctly classified samples per class
correct_mask = (preds_test == y_te)
selected = {}
for cat in ['Normal','DoS','Fuzzers']:
    candidates = np.where((test_cats == cat) & correct_mask)[0]
    if cat != 'Normal':
        candidates = np.array([i for i in candidates if y_te[i] == 1])
    n = min(N_SAMPLES, len(candidates))
    selected[cat] = np.random.RandomState(SEED).choice(candidates, n, replace=False).tolist()
    print(f'{cat}: {len(selected[cat])} correctly classified samples')

# Hook ALL attention layers in ALL views
attn_data = {}  # {view_name: {layer_name: np.array(n_samples, seq_len, seq_len)}}
for vi, vt in enumerate(model.encoder.view_transformers):
    vn = f'View_{vi+1}'; attn_data[vn] = {}
    for li, layer in enumerate(vt.blocks):
        ln = f'Layer_{li+1}'; attn_data[vn][ln] = []
        orig = layer.self_attn.forward
        def make_patch(vname, lname, original):
            def patched(self, q, k, v, **kw):
                kw['need_weights']=True; kw['average_attn_weights']=True
                out, w = original(q, k, v, **kw)
                attn_data[vname][lname].append(w.detach().cpu().numpy()[0])
                return out, w
            return patched
        layer.self_attn.forward = types.MethodType(make_patch(vn, ln, orig), layer.self_attn)

# Forward all selected samples
all_samples = []
for cat in ['Normal','DoS','Fuzzers']:
    for idx in selected[cat]:
        all_samples.append(idx)
        with torch.no_grad():
            _ = model(X_te_t[idx:idx+1].to(DEVICE))

total_samples = len(all_samples)
print(f'Total forward passes: {total_samples}')

# Convert attention lists to arrays
for vn in attn_data:
    for ln in attn_data[vn]:
        attn_data[vn][ln] = np.array(attn_data[vn][ln])

# ═══════ STEP 4: Compute Statistics ═══════
print(); print('='*60); print('STEP 4: Computing attention statistics'); print('='*60)

offsets = {'Normal':0, 'DoS':len(selected['Normal']), 'Fuzzers':len(selected['Normal'])+len(selected['DoS'])}
class_order = ['Normal', 'DoS', 'Fuzzers']

# For each (view, layer, class): mean attention per feature
stats_data = {}  # {view: {layer: {class: {'mean':(seq,), 'std':(seq,)}}}}

for vn in ['View_1','View_2','View_3']:
    stats_data[vn] = {}
    for ln in ['Layer_1','Layer_2','Layer_3','Layer_4']:
        stats_data[vn][ln] = {}
        attns = attn_data[vn][ln]  # (total_samples, seq, seq)
        
        for cat in class_order:
            start = offsets[cat]; end = start + len(selected[cat])
            cat_attns = attns[start:end]  # (n_samples, seq, seq)
            # Average attention received by each feature (column-wise mean)
            feat_attn = cat_attns.mean(axis=1)  # (n_samples, seq)
            stats_data[vn][ln][cat] = {
                'mean': feat_attn.mean(axis=0),  # (seq,)
                'std': feat_attn.std(axis=0, ddof=1)  # (seq,)
            }

# Cross-class statistical tests for last layer
print('Cross-class t-tests (DoS vs Normal, top features):')
for vn in ['View_1','View_2','View_3']:
    ln = 'Layer_4'
    attns = attn_data[vn][ln]
    normal_feat = attns[:len(selected['Normal'])].mean(axis=1)  # (N_n, seq)
    dos_feat = attns[offsets['DoS']:offsets['DoS']+len(selected['DoS'])].mean(axis=1)  # (N_d, seq)
    
    t_stats, p_vals = stats.ttest_ind(dos_feat, normal_feat, axis=0)
    sig_feats = np.where(p_vals < 0.01)[0]
    print(f'{vn}: {len(sig_feats)}/{42} features significantly different (p<0.01)')
    if len(sig_feats) > 0:
        print(f'  Top indices: {sig_feats[:10]}')

# ═══════ STEP 5: Publication-Quality Visualizations ═══════
print(); print('='*60); print('STEP 5: Generating figures'); print('='*60)

feat_names_view1 = ['dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sinpkt','dinpkt']  # 15
feat_names_view2 = ['sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','res_bdy_len','ct_srv_src']  # 14
feat_names_view3 = ['ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ ltm','ct_srv_dst','is_sm_ips_ports','ct_state_ttl']  # 13
view_feat_names = {'View_1': feat_names_view1, 'View_2': feat_names_view2, 'View_3': feat_names_view3}

# FIGURE 1: Per-class average attention heatmaps (last layer, 3 views x 3 classes = 9 panels)
fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle('Average Attention Patterns by View and Attack Category (Last Layer)', fontsize=15, fontweight='bold')

for ri, cat in enumerate(class_order):
    for ci, vn in enumerate(['View_1','View_2','View_3']):
        ax = axes[ri, ci]
        s = stats_data[vn]['Layer_4'][cat]
        im = ax.imshow(s['mean'].reshape(-1,1), cmap='YlOrRd', aspect='auto')
        nf = len(s['mean'])
        names = view_feat_names.get(vn, [f'F{i+1}' for i in range(nf)])
        ax.set_yticks(range(nf)); ax.set_yticklabels(names[:nf], fontsize=6)
        ax.set_xlim(-1,1); ax.set_xticks([])
        ax.set_title(f'{cat} — {vn.replace(chr(95),chr(32))}', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
        # Mark top-3 attended features
        top3 = np.argsort(s['mean'])[-3:][::-1]
        for ti in top3:
            ax.annotate('*', xy=(0.5, ti), ha='center', fontsize=10, color='blue', fontweight='bold')

plt.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT, 'fig1_attention_heatmaps.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig1: per-class heatmaps')

# FIGURE 2: Feature-level attention comparison (bar chart with error bars, View 1 only for clarity)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Feature-level Attention: Normal vs DoS vs Fuzzers (View 1: Flow Statistics)', fontsize=14, fontweight='bold')

for vi, vn in enumerate(['View_1','View_2','View_3']):
    ax = axes[vi]
    s_list = [stats_data[vn]['Layer_4'][cat] for cat in class_order]
    nf = len(s_list[0]['mean'])
    x = np.arange(nf); w = 0.25
    colors = ['#3498DB','#E74C3C','#2ECC71']
    for si, (cat, color) in enumerate(zip(class_order, colors)):
        ax.bar(x + (si-1)*w, s_list[si]['mean'], w, yerr=s_list[si]['std'], 
               label=cat, color=color, alpha=0.7, capsize=2, error_kw={'linewidth':0.5})
    names = view_feat_names.get(vn, [f'F{i+1}' for i in range(nf)])
    ax.set_xticks(x); ax.set_xticklabels(names[:nf], rotation=45, ha='right', fontsize=7)
    ax.set_title(vn.replace(chr(95),chr(32)), fontsize=10)
    ax.set_ylabel('Avg Attention'); ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_attention_barchart.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig2: feature bar charts')

# FIGURE 3: Attention concentration (Gini coefficient per class)
fig, ax = plt.subplots(figsize=(8, 5))
for vn in ['View_1','View_2','View_3']:
    for cat, marker in zip(class_order, ['o','s','^']):
        s = stats_data[vn]['Layer_4'][cat]
        # Sort and compute cumulative
        sorted_attn = np.sort(s['mean'])[::-1]
        cumsum = np.cumsum(sorted_attn) / np.sum(sorted_attn)
        ax.plot(np.arange(1, len(cumsum)+1)/len(cumsum)*100, cumsum*100, 
                marker=marker, markersize=3, label=f'{vn}:{cat}', linewidth=1.5)

ax.set_xlabel('Cumulative Features (%)'); ax.set_ylabel('Cumulative Attention (%)')
ax.set_title('Attention Concentration: How Many Features Capture 80% of Attention?', fontsize=13)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_attention_concentration.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig3: concentration curves')

# FIGURE 4: Layer-wise attention evolution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Attention Evolution Across Transformer Layers (View 1)', fontsize=14, fontweight='bold')

for ci, cat in enumerate(class_order):
    ax = axes[ci]
    nf = len(stats_data['View_1']['Layer_4'][cat]['mean'])
    x = np.arange(nf)
    for li, ln in enumerate(['Layer_1','Layer_2','Layer_3','Layer_4']):
        s = stats_data['View_1'][ln][cat]
        ax.plot(x, s['mean'], 'o-', markersize=3, linewidth=1.5, label=ln, alpha=0.8)
    ax.set_title(f'{cat}', fontsize=11)
    ax.set_xlabel('Feature index'); ax.set_ylabel('Avg Attention')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_layer_evolution.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig4: layer evolution')

# FIGURE 5: Statistical significance map (DoS vs Normal, all views)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
fig.suptitle('Statistical Significance: DoS vs Normal Attention Differences (p-values)', fontsize=13, fontweight='bold')

attns_v1 = attn_data['View_1']['Layer_4']
attns_v2 = attn_data['View_2']['Layer_4']
attns_v3 = attn_data['View_3']['Layer_4']
for vi, (attns, vn, ax) in enumerate(zip([attns_v1, attns_v2, attns_v3], 
                                          ['View_1','View_2','View_3'], axes)):
    normal_feat = attns[:len(selected['Normal'])].mean(axis=1)
    dos_feat = attns[offsets['DoS']:offsets['DoS']+len(selected['DoS'])].mean(axis=1)
    _, p_vals = stats.ttest_ind(dos_feat, normal_feat, axis=0)
    
    nf = len(p_vals)
    sig_map = np.zeros((nf, 1))
    sig_map[p_vals < 0.05] = 0.5; sig_map[p_vals < 0.01] = 1.0
    ax.imshow(sig_map, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(nf))
    names = view_feat_names.get(vn, [f'F{i+1}' for i in range(nf)])
    ax.set_yticklabels(names[:nf], fontsize=6)
    ax.set_xticks([])
    ax.set_title(f'{vn} (sig features: {np.sum(p_vals<0.01)}/p<0.01, {np.sum(p_vals<0.05)}/p<0.05)', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_significance_map.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig5: significance map')

# Summary stats JSON
summary = {}
for vn in ['View_1','View_2','View_3']:
    summary[vn] = {}
    for ln in ['Layer_4']:
        summary[vn][ln] = {}
        attns = attn_data[vn][ln]
        for cat in class_order:
            start = offsets[cat]; end = start + len(selected[cat])
            cat_attns = attns[start:end].mean(axis=1).mean(axis=0)
            top5_idx = np.argsort(cat_attns)[-5:][::-1].tolist()
            top5_vals = cat_attns[top5_idx].tolist()
            names = view_feat_names.get(vn, [f'F{i}' for i in range(len(cat_attns))])
            summary[vn][ln][cat] = {
                'concentration': float(np.sort(cat_attns)[-5:].sum() / cat_attns.sum()),
                'top5_features': [names[i] for i in top5_idx],
                'top5_values': [round(v, 4) for v in top5_vals]
            }

with open(os.path.join(OUT, 'attention_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nDone. {len(os.listdir(OUT))} files in {OUT}')

# ═══════ STEP 6: Individual Sample Heatmaps ═══════
print(); print('='*60); print('STEP 6: Individual sample heatmaps'); print('='*60)

# FIGURE 6: Sample diversity grid (25 samples × 3 classes × 1 view = 75 panels)
for target_view in ['View_1']:  # Focus on View 1 for clarity
    fig, axes = plt.subplots(3, 25, figsize=(40, 5))
    fig.suptitle(f'Sample-Level Attention Diversity: 25 Individual {target_view} Heatmaps per Class ({N_SAMPLES} total each)', 
                 fontsize=16, fontweight='bold')
    
    attns_v = attn_data[target_view]['Layer_4']
    for ri, cat in enumerate(class_order):
        start = offsets[cat]; end = start + len(selected[cat])
        cat_attns = attns_v[start:end]  # (N, seq, seq)
        
        # Select 25 evenly spaced samples
        indices_25 = np.linspace(0, len(cat_attns)-1, 25, dtype=int)
        
        for ci, idx in enumerate(indices_25):
            ax = axes[ri, ci]
            sample_attn = cat_attns[idx].mean(axis=1).reshape(-1, 1)  # average incoming attention
            ax.imshow(sample_attn, cmap='YlOrRd', aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0: ax.set_ylabel(cat, fontsize=10, fontweight='bold')
            if ri == 0: ax.set_title(f'S{idx}', fontsize=8)
    
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(OUT, 'fig6_sample_diversity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] fig6: 75 individual heatmaps ({target_view})')

# FIGURE 7: Extreme case comparison (most peaked vs most dispersed per class)
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Extreme Case Analysis: Most Peaked vs Most Dispersed Attention per Class', fontsize=14, fontweight='bold')

for ri, cat in enumerate(class_order):
    start = offsets[cat]; end = start + len(selected[cat])
    cat_attns = attn_data['View_1']['Layer_4'][start:end]
    
    # Compute concentration (Gini-like: fraction of attention in top-5 features)
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
        ax.set_yticks(range(nf))
        names = view_feat_names.get('View_1', [f'F{i+1}' for i in range(nf)])
        ax.set_yticklabels(names[:nf], fontsize=7)
        ax.set_xticks([])
        ax.set_title(f'{cat} — {label}\nTop-5 conc: {concentrations[idx]:.2f}', fontsize=10)

plt.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT, 'fig7_extreme_cases.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig7: extreme cases')

# FIGURE 8: Attention variance (std across samples) per feature per class
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Cross-Sample Attention Variance: How Consistent is Each Feature? (View 1)', fontsize=14, fontweight='bold')

for vi, (vn, ax) in enumerate(zip(['View_1','View_2','View_3'], axes)):
    nf = len(stats_data[vn]['Layer_4']['Normal']['std'])
    x = np.arange(nf)
    for cat, color in zip(class_order, ['#3498DB','#E74C3C','#2ECC71']):
        s = stats_data[vn]['Layer_4'][cat]
        cv = s['std'] / (s['mean'] + 1e-8)  # coefficient of variation
        ax.bar(x + (class_order.index(cat)-1)*0.25, cv, 0.25, label=cat, color=color, alpha=0.7)
    names = view_feat_names.get(vn, [f'F{i}' for i in range(nf)])
    ax.set_xticks(x); ax.set_xticklabels(names[:nf], rotation=45, ha='right', fontsize=6)
    ax.set_title(vn.replace(chr(95),chr(32)), fontsize=10)
    ax.set_ylabel('CV (std/mean)'); ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig8_attention_variance.png'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] fig8: attention variance')

# Update summary
summary['sample_sizes'] = {cat: len(selected[cat]) for cat in class_order}
with open(os.path.join(OUT, 'attention_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nALL DONE. {len(os.listdir(OUT))} files in {OUT}')

