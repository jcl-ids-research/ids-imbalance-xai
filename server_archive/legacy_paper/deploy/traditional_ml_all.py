"""
Traditional ML baselines: RF + XGBoost + MLP on ALL 4 datasets.
UNSW-NB15, CIC-IDS-2017, CIC-DDoS2019, NSL-KDD.
Outputs full metrics (Acc/Prec/Rec/F1) and confusion matrices.
"""
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OUT = '/opt/ids_revision/results/traditional_ml'
os.makedirs(OUT, exist_ok=True)

def plot_cm(cm, class_names, title, path):
    fig,ax=plt.subplots(figsize=(6,5) if len(class_names)<=5 else (9,8))
    im=ax.imshow(cm,cmap='Blues')
    n=len(class_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names,rotation=45,ha='right',fontsize=7)
    ax.set_yticklabels(class_names,fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title,fontsize=10,fontweight='bold')
    for i in range(n):
        for j in range(n):
            if cm[i,j]>0:
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6,
                       color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.colorbar(im,ax=ax,shrink=0.8); plt.tight_layout()
    fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close()

def eval_model(y_true, y_pred):
    acc=accuracy_score(y_true,y_pred)
    p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,average='weighted',zero_division=0)
    cm=confusion_matrix(y_true,y_pred)
    return {'acc':float(acc),'prec':float(p),'rec':float(r),'f1':float(f1),'cm':cm.tolist()}

# ═══ UNSW-NB15 ═══
def load_unsw(multiclass=False):
    tr=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
    te=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
    df=pd.concat([tr,te],ignore_index=True)
    df=df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns],errors='ignore')
    if multiclass:
        yr=df['attack_cat'].astype(str).str.strip().replace('','Normal')
        le=LabelEncoder(); y=le.fit_transform(yr); cls=list(le.classes_)
    else: y=df['label'].values.astype(np.int64); cls=['Normal','Attack']
    feats=[c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
    X=df[feats]
    for c in X.select_dtypes(include=['object']).columns: X[c]=LabelEncoder().fit_transform(X[c].astype(str))
    X=X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X=StandardScaler().fit_transform(X)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_tr,X_te,y_tr,y_te,cls

# ═══ CIC ═══
sys.path.insert(0,'/opt/ids_revision/deploy')
from cic_data_loader import load_cicflowmeter

def load_cic(path,max_s=60000):
    d=load_cicflowmeter(path,rows_per_file=20000,max_total=max_s,binary=True)
    X=np.concatenate([d['X_train'][:40000].numpy(),d['X_val'][:5000].numpy(),d['X_test'][:15000].numpy()])
    y=np.concatenate([d['y_train'][:40000].numpy(),d['y_val'][:5000].numpy(),d['y_test'][:15000].numpy()])
    n=int(len(X)*0.7)
    return X[:n],X[n:],y[:n],y[n:],['Normal','Attack']

# ═══ NSL-KDD ═══
COL=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
     'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
     'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
     'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
     'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
     'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
     'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']

def load_nsl(multiclass=False):
    df_tr=pd.read_csv('/opt/NSL-KDD/KDDTrain+.txt',names=COL)
    df_te=pd.read_csv('/opt/NSL-KDD/KDDTest+.txt',names=COL)
    df=pd.concat([df_tr,df_te],ignore_index=True)
    for c in ['protocol_type','service','flag']: df[c]=LabelEncoder().fit_transform(df[c].astype(str))
    if multiclass:
        attack_map={'normal':'normal','back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS','teardrop':'DoS',
                    'satan':'Probe','ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
                    'guess_passwd':'R2L','ftp_write':'R2L','imap':'R2L','phf':'R2L','multihop':'R2L','warezmaster':'R2L','warezclient':'R2L','spy':'R2L',
                    'buffer_overflow':'U2R','loadmodule':'U2R','rootkit':'U2R','perl':'U2R'}
        yr=df['label'].map(attack_map).fillna('normal')
        y=LabelEncoder().fit_transform(yr); cls=list(LabelEncoder().fit(yr).classes_)
    else: y=(df['label']!='normal').astype(int).values; cls=['Normal','Attack']
    feats=[c for c in COL if c not in ['label','difficulty']]
    X=df[feats].fillna(0).astype(np.float32)
    X=StandardScaler().fit_transform(X)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_tr,X_te,y_tr,y_te,cls

# ═══ RUN ALL ═══
all_results={}
configs=[
    ('UNSW-NB15','binary',lambda:load_unsw(False)),
    ('CIC-IDS-2017','binary',lambda:load_cic('/opt/CIC-IDS-2017/MachineLearningCVE',60000)),
    ('CIC-DDoS2019','binary',lambda:load_cic('/opt/CIC-DDoS2019/all',60000)),
    ('NSL-KDD','binary',lambda:load_nsl(False)),
]

t0=time.time()
for ds,task,loader in configs:
    X_tr,X_te,y_tr,y_te,cls=loader()
    nc=len(cls)
    print(f'\n{ds} ({len(X_tr)} train, {len(X_te)} test, {nc} classes)')
    all_results[ds]={}
    
    # RF
    t1=time.time()
    rf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
    rf.fit(X_tr,y_tr); preds_rf=rf.predict(X_te)
    all_results[ds]['RF']=eval_model(y_te,preds_rf)
    plot_cm(confusion_matrix(y_te,preds_rf),cls,f'{ds} - RF',os.path.join(OUT,f'cm_{ds}_RF.png'))
    print(f'  RF: Acc={all_results[ds]["RF"]["acc"]:.4f}, F1={all_results[ds]["RF"]["f1"]:.4f}, Time={time.time()-t1:.1f}s')
    
    # XGBoost
    t1=time.time()
    import xgboost as xgb
    xgb_m=xgb.XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=42,use_label_encoder=False,eval_metric='mlogloss')
    xgb_m.fit(X_tr,y_tr); preds_xgb=xgb_m.predict(X_te)
    all_results[ds]['XGBoost']=eval_model(y_te,preds_xgb)
    plot_cm(confusion_matrix(y_te,preds_xgb),cls,f'{ds} - XGBoost',os.path.join(OUT,f'cm_{ds}_XGBoost.png'))
    print(f'  XGB: Acc={all_results[ds]["XGBoost"]["acc"]:.4f}, F1={all_results[ds]["XGBoost"]["f1"]:.4f}, Time={time.time()-t1:.1f}s')
    
    # MLP
    t1=time.time()
    X_tr_t=torch.FloatTensor(X_tr); y_tr_t=torch.LongTensor(y_tr); X_te_t=torch.FloatTensor(X_te)
    mlp=nn.Sequential(nn.Linear(X_tr.shape[1],256),nn.ReLU(),nn.Dropout(0.3),
                       nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),
                       nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.3),
                       nn.Linear(64,nc)).cuda()
    opt=torch.optim.Adam(mlp.parameters(),lr=0.001)
    n=len(X_tr_t)
    for ep in range(30):
        mlp.train(); perm=torch.randperm(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]; bx,by=X_tr_t[idx].cuda(),y_tr_t[idx].cuda()
            opt.zero_grad(); loss=nn.CrossEntropyLoss()(mlp(bx),by)
            loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad(): preds_mlp=mlp(X_te_t.cuda()).argmax(dim=1).cpu().numpy()
    all_results[ds]['MLP']=eval_model(y_te,preds_mlp)
    plot_cm(confusion_matrix(y_te,preds_mlp),cls,f'{ds} - MLP',os.path.join(OUT,f'cm_{ds}_MLP.png'))
    print(f'  MLP: Acc={all_results[ds]["MLP"]["acc"]:.4f}, F1={all_results[ds]["MLP"]["f1"]:.4f}, Time={time.time()-t1:.1f}s')

# ═══ COMPARISON CHART ═══
our_f1 = {'UNSW-NB15':93.63,'CIC-IDS-2017':99.11,'CIC-DDoS2019':99.95,'NSL-KDD':99.18}
fig,axes=plt.subplots(1,4,figsize=(20,5))
for i,(ds,f1s) in enumerate([(ds,[all_results[ds][m]['f1']*100 for m in ['RF','XGBoost','MLP']]) for ds in ['UNSW-NB15','CIC-IDS-2017','CIC-DDoS2019','NSL-KDD']]):
    ax=axes[i]; models=['RF','XGBoost','MLP','Ours']; vals=f1s+[our_f1[ds]]
    colors=['#F39C12','#E74C3C','#3498DB','#2E86AB']
    ax.bar(models,vals,color=colors,edgecolor='white')
    for j,v in enumerate(vals): ax.text(j,v+1,f'{v:.1f}',ha='center',fontsize=9,fontweight='bold')
    ax.set_title(ds,fontsize=11,fontweight='bold'); ax.set_ylabel('F1 (%)'); ax.set_ylim(40,110)
plt.tight_layout()
fig.savefig(os.path.join(OUT,'traditional_ml_comparison.png'),dpi=300,bbox_inches='tight'); plt.close()
print('\nComparison chart saved.')

with open(os.path.join(OUT,'traditional_ml_results.json'),'w') as f: json.dump(all_results,f,indent=2)
print(f'Done. Time: {(time.time()-t0)/60:.0f}min. {len(os.listdir(OUT))} files.')
