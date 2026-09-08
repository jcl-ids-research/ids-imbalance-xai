"""CNN baseline on all 4 datasets. Reuse XGB/MLP from previous run."""
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OUT = '/opt/ids_revision/results/traditional_ml'
DEVICE = torch.device('cuda')

# ═══ CNN Model ═══
class CNN1D(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(256),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.cls = nn.Linear(256, n_classes)
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, F)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)
        return self.cls(x)

def plot_cm(cm, class_names, title, path):
    fig,ax=plt.subplots(figsize=(6,5) if len(class_names)<=5 else (9,8))
    im=ax.imshow(cm,cmap='Blues'); n=len(class_names)
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

# ═══ Data loaders ═══
def load_unsw():
    tr=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
    te=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
    df=pd.concat([tr,te],ignore_index=True)
    df=df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns],errors='ignore')
    y=df['label'].values.astype(np.int64); cls=['Normal','Attack']
    feats=[c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
    X=df[feats]
    for c in X.select_dtypes(include=['object']).columns: X[c]=LabelEncoder().fit_transform(X[c].astype(str))
    X=X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X=StandardScaler().fit_transform(X)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_tr,X_te,y_tr,y_te,cls

sys.path.insert(0,'/opt/ids_revision/deploy')
from cic_data_loader import load_cicflowmeter
def load_cic(path,mx=60000):
    d=load_cicflowmeter(path,rows_per_file=20000,max_total=mx,binary=True)
    X=np.concatenate([d['X_train'][:40000].numpy(),d['X_val'][:5000].numpy(),d['X_test'][:15000].numpy()])
    y=np.concatenate([d['y_train'][:40000].numpy(),d['y_val'][:5000].numpy(),d['y_test'][:15000].numpy()])
    n=int(len(X)*0.7); return X[:n],X[n:],y[:n],y[n:],['Normal','Attack']

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
    return X_tr,X_te,y_tr,y_te,cls

# ═══ Run CNN on all datasets ═══
configs=[
    ('UNSW-NB15',load_unsw),
    ('CIC-IDS-2017',lambda:load_cic('/opt/CIC-IDS-2017/MachineLearningCVE')),
    ('CIC-DDoS2019',lambda:load_cic('/opt/CIC-DDoS2019/all')),
    ('NSL-KDD',load_nsl),
]

cnn_results={}
t0=time.time()
for ds,loader in configs:
    X_tr,X_te,y_tr,y_te,cls=loader()
    X_tr_t=torch.FloatTensor(X_tr); y_tr_t=torch.LongTensor(y_tr); X_te_t=torch.FloatTensor(X_te)
    nc=len(cls)
    print(f'{ds}: train={X_tr.shape}, test={X_te.shape}, classes={nc}')
    
    t1=time.time()
    m=CNN1D(X_tr.shape[1],nc).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=0.001)
    n=len(X_tr_t)
    for ep in range(30):
        m.train(); perm=torch.randperm(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]; bx,by=X_tr_t[idx].to(DEVICE),y_tr_t[idx].to(DEVICE)
            opt.zero_grad(); loss=nn.CrossEntropyLoss()(m(bx),by)
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): preds=m(X_te_t.to(DEVICE)).argmax(dim=1).cpu().numpy()
    acc=accuracy_score(y_te,preds)
    p,r,f1,_=precision_recall_fscore_support(y_te,preds,average='weighted',zero_division=0)
    cm=confusion_matrix(y_te,preds)
    cnn_results[ds]={'acc':float(acc),'prec':float(p),'rec':float(r),'f1':float(f1)}
    plot_cm(cm,cls,f'{ds} - CNN',os.path.join(OUT,f'cm_{ds}_CNN.png'))
    print(f'  CNN: Acc={acc:.4f}, F1={f1:.4f}, Time={time.time()-t1:.1f}s')

# Save CNN results
import json
with open(os.path.join(OUT,'cnn_results.json'),'w') as f: json.dump(cnn_results,f,indent=2)

# ═══ Generate comparison chart (CNN + XGB + MLP + Ours) ═══
old=json.load(open(os.path.join(OUT,'traditional_ml_results.json')))
our={'UNSW-NB15':93.63,'CIC-IDS-2017':99.11,'CIC-DDoS2019':99.95,'NSL-KDD':99.18}
datasets=['UNSW-NB15','CIC-IDS-2017','CIC-DDoS2019','NSL-KDD']

fig,axes=plt.subplots(1,4,figsize=(22,5.5))
fig.suptitle('CNN, XGBoost, MLP vs Our Model (Binary F1)',fontsize=14,fontweight='bold')
colors=['#9B59B6','#F39C12','#E74C3C','#3498DB','#2E86AB']

for i,ds in enumerate(datasets):
    ax=axes[i]
    vals=[cnn_results[ds]['f1']*100, old[ds]['XGBoost']['f1']*100, old[ds]['MLP']['f1']*100, our[ds]]
    labels=['CNN','XGBoost','MLP','Ours']
    ax.bar(labels,vals,color=colors[1:],edgecolor='white')
    for j,v in enumerate(vals): ax.text(j,v+1,f'{v:.1f}',ha='center',fontsize=9,fontweight='bold')
    ax.set_title(ds,fontsize=11,fontweight='bold'); ax.set_ylabel('F1 (%)')
    ax.set_ylim(min(vals)-8,max(vals)+8)
plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(os.path.join(OUT,'cnn_xgb_mlp_comparison.png'),dpi=300,bbox_inches='tight'); plt.close()
print('\nComparison chart saved.')
print(f'Done. Time: {(time.time()-t0)/60:.0f}min')
