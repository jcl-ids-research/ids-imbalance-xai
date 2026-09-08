"""
Ablation study on CIC-IDS-2017, CIC-DDoS2019, NSL-KDD.
4 variants per dataset: full, wo_diffusion, wo_multiview, wo_both.
All run overnight. ~10 hours total.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'; os.environ['OMP_NUM_THREADS']='4'; os.environ['MKL_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '/opt/ids_revision')
sys.path.insert(0, '/opt/ids_revision/deploy')
from ablation.models.model_variants import create_variant, count_parameters
from cic_data_loader import load_cicflowmeter

DEVICE = torch.device('cuda')
OUT = '/opt/ids_revision/results/ablation_cross_dataset'
os.makedirs(OUT, exist_ok=True)
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

CONFIG = {
    'd_model': 128, 'nhead': 8, 'num_layers': 4, 'n_views': 3,
    'fusion_method': 'concat', 'dim_feedforward': 512, 'dropout': 0.1,
    'learning_rate': 1e-4, 'weight_decay': 1e-5, 'batch_size': 512, 'epochs': 50, 'patience': 10,
}

def train_variant(model, X_tr, y_tr, X_vl, y_vl, X_te, y_te, epochs=50):
    model = model.to(DEVICE)
    tl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=CONFIG['batch_size'], shuffle=True)
    vl = DataLoader(TensorDataset(X_vl, y_vl), batch_size=CONFIG['batch_size'])
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_f1, best_state, patience = 0, None, 0
    t0 = time.time()
    
    for ep in range(1, epochs+1):
        model.train()
        for bx, by in tl:
            opt.zero_grad(); l = crit(model(bx.to(DEVICE)), by.to(DEVICE))
            l.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval(); ap, al = [], []
        with torch.no_grad():
            for bx, by in vl:
                preds = model(bx.to(DEVICE)).argmax(dim=1)
                ap.extend(preds.cpu().numpy()); al.extend(by.numpy())
        _, _, f1, _ = precision_recall_fscore_support(al, ap, average='weighted', zero_division=0)
        if f1 > best_f1: best_f1 = f1; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; patience = 0
        else: patience += 1
        if patience >= CONFIG['patience']: break
    
    if best_state: model.load_state_dict(best_state)
    model.eval(); ap, al = [], []
    with torch.no_grad():
        for bx, by in DataLoader(TensorDataset(X_te, y_te), batch_size=CONFIG['batch_size']):
            preds = model(bx.to(DEVICE)).argmax(dim=1)
            ap.extend(preds.cpu().numpy()); al.extend(by.numpy())
    acc = accuracy_score(al, ap)
    p, r, f1, _ = precision_recall_fscore_support(al, ap, average='weighted', zero_division=0)
    f1_m, _, _, _ = precision_recall_fscore_support(al, ap, average='macro', zero_division=0)
    n_params = count_parameters(model)
    elapsed = time.time() - t0
    return {'acc': float(acc), 'f1_w': float(f1), 'f1_m': float(f1_m), 'prec': float(p), 'rec': float(r),
            'params': n_params, 'time_sec': elapsed, 'epochs': ep}

# ═══ CIC ═══
def load_cic_dataset(path):
    d = load_cicflowmeter(path, rows_per_file=50000, max_total=400000, binary=True)
    X_tr = d['X_train']; y_tr = d['y_train']
    X_vl = d['X_val']; y_vl = d['y_val']
    X_te = d['X_test']; y_te = d['y_test']
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te, d['input_dim'], d['num_classes']

# ═══ NSL-KDD ═══
COL=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
     'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
     'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
     'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
     'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
     'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
     'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']

def load_nsl():
    df_tr=pd.read_csv('/opt/NSL-KDD/KDDTrain+.txt',names=COL)
    df_te=pd.read_csv('/opt/NSL-KDD/KDDTest+.txt',names=COL)
    df=pd.concat([df_tr,df_te],ignore_index=True)
    for c in ['protocol_type','service','flag']: df[c]=LabelEncoder().fit_transform(df[c].astype(str))
    y=(df['label']!='normal').astype(int).values; cls=['Normal','Attack']
    feats=[c for c in COL if c not in ['label','difficulty']]
    X=df[feats].fillna(0).astype(np.float32)
    X=StandardScaler().fit_transform(X)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    X_tr,X_vl,y_tr,y_vl=train_test_split(X_tr,y_tr,test_size=0.1,random_state=42)
    return (torch.FloatTensor(X_tr),torch.LongTensor(y_tr),torch.FloatTensor(X_vl),torch.LongTensor(y_vl),
            torch.FloatTensor(X_te),torch.LongTensor(y_te),X.shape[1],2)

# ═══ RUN ═══
datasets = [
    ('CIC-IDS-2017', lambda: load_cic_dataset('/opt/CIC-IDS-2017/MachineLearningCVE')),
    ('CIC-DDoS2019', lambda: load_cic_dataset('/opt/CIC-DDoS2019/all')),
    ('NSL-KDD', load_nsl),
]

variants = ['full', 'wo_diffusion', 'wo_multiview', 'wo_both']
all_results = {}
t_total = time.time()

for ds_name, loader in datasets:
    X_tr, y_tr, X_vl, y_vl, X_te, y_te, input_dim, n_classes = loader()
    print(f'\n{"="*60}\n{ds_name}: {X_tr.shape[0]} train, {X_te.shape[0]} test, {input_dim} dims, {n_classes} classes\n{"="*60}')
    all_results[ds_name] = {}
    
    for variant in variants:
        print(f'  Training {variant}...', flush=True)
        t1 = time.time()
        model = create_variant(variant, input_dim=input_dim, num_classes=n_classes, **{k:CONFIG[k] for k in ['d_model','nhead','num_layers','n_views','fusion_method','dim_feedforward','dropout']})
        r = train_variant(model, X_tr, y_tr, X_vl, y_vl, X_te, y_te, CONFIG['epochs'])
        r['variant'] = variant
        all_results[ds_name][variant] = r
        print(f'    Acc={r["acc"]:.4f}, F1(w)={r["f1_w"]:.4f}, Params={r["params"]:,}, Time={r["time_sec"]/60:.1f}min', flush=True)
        del model; torch.cuda.empty_cache()
    
    # Print summary for this dataset
    print(f'\n{ds_name} Summary:')
    print(f'  {"Variant":20s} {"Params":>10s} {"Acc":>8s} {"F1(w)":>8s}')
    for v in variants:
        r = all_results[ds_name][v]
        print(f'  {v:20s} {r["params"]:>10,d} {r["acc"]:>8.4f} {r["f1_w"]:>8.4f}')

with open(os.path.join(OUT, 'cross_dataset_ablation.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\nDone. Total: {(time.time()-t_total)/3600:.1f}h')
