"""
ALL confusion matrices on ALL 4 datasets for ALL models.
UNSW-NB15, CIC-IDS-2017, CIC-DDoS2019, NSL-KDD.
Models: Full, FeatureTransformer, GraphSAGE.
~20 min on A100.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'; os.environ['OMP_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import kneighbors_graph
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

sys.path.insert(0, '/opt/ids_revision'); sys.path.insert(0, '/opt/ids_revision/deploy')
from ablation.models.feature_transformer import FeatureTransformer
from ablation.models.model_variants import create_variant
from torch_geometric.nn import SAGEConv
from cic_data_loader import load_cicflowmeter

DEVICE = torch.device('cuda')
OUT = '/opt/ids_revision/results/confusion_matrices'
os.makedirs(OUT, exist_ok=True)

class GraphSAGE(nn.Module):
    def __init__(self, in_c, hid, out_c, layers=3, dropout=0.3):
        super().__init__()
        self.convs=nn.ModuleList([SAGEConv(in_c,hid)])
        for _ in range(layers-2): self.convs.append(SAGEConv(hid,hid))
        self.convs.append(SAGEConv(hid,hid))
        self.dropout=dropout; self.cls=nn.Linear(hid,out_c)
    def forward(self,x,ei):
        for c in self.convs: x=F.relu(c(x,ei)); x=F.dropout(x,p=self.dropout,training=self.training)
        return self.cls(x)

def plot_cm(cm, class_names, title, path):
    fig,ax=plt.subplots(figsize=(8,7) if len(class_names)<=5 else (10,9))
    im=ax.imshow(cm,cmap='Blues')
    n=len(class_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names,rotation=45,ha='right',fontsize=7)
    ax.set_yticklabels(class_names,fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title,fontsize=11,fontweight='bold')
    for i in range(n):
        for j in range(n):
            if cm[i,j]>0:
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=5,
                       color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.colorbar(im,ax=ax,shrink=0.8)
    plt.tight_layout(); fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close()

def train_feattrans(X_tr,y_tr,X_te,y_te,n_classes,epochs=15):
    m=FeatureTransformer(n_features=X_tr.shape[1],num_classes=n_classes,d_model=64,nhead=8,num_layers=4,pooling='mean').to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=0.0001)
    n=len(X_tr)
    for ep in range(1,epochs+1):
        m.train(); perm=torch.randperm(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]; bx,by=X_tr[idx].to(DEVICE),y_tr[idx].to(DEVICE)
            opt.zero_grad(); l=nn.CrossEntropyLoss()(m(bx),by)
            l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval()
    with torch.no_grad(): preds=m(X_te.to(DEVICE)).argmax(dim=1).cpu()
    return confusion_matrix(y_te,preds),accuracy_score(y_te,preds)

def train_gnn(X_tr,y_tr,X_te,y_te,n_classes,epochs=30):
    X_all=torch.cat([X_tr[:25000],X_te[:15000]],dim=0)
    y_all=torch.cat([y_tr[:25000],y_te[:15000]],dim=0)
    k=min(5,X_all.shape[0]-1)
    adj=kneighbors_graph(X_all.numpy(),n_neighbors=k,mode='connectivity',include_self=False)
    ei=torch.tensor(np.array(adj.nonzero()),dtype=torch.long)
    m=GraphSAGE(X_all.shape[1],128,n_classes).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=0.001,weight_decay=5e-4)
    for ep in range(1,epochs+1):
        m.train(); opt.zero_grad()
        l=nn.CrossEntropyLoss()(m(X_all.to(DEVICE),ei.to(DEVICE))[:25000],y_all[:25000].to(DEVICE))
        l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval()
    with torch.no_grad(): preds=m(X_all.to(DEVICE),ei.to(DEVICE))[25000:].argmax(dim=1).cpu()
    return confusion_matrix(y_all[25000:],preds),accuracy_score(y_all[25000:],preds)

def train_full(X_tr,y_tr,X_te,y_te,n_classes,epochs=15):
    m=create_variant('full',input_dim=X_tr.shape[1],num_classes=n_classes,d_model=128,nhead=8,
                      num_layers=4,n_views=3,fusion_method='concat',dim_feedforward=512,dropout=0.1).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    n=len(X_tr)
    for ep in range(1,epochs+1):
        m.train(); perm=torch.randperm(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]; bx,by=X_tr[idx].to(DEVICE),y_tr[idx].to(DEVICE)
            opt.zero_grad(); l=nn.CrossEntropyLoss()(m(bx),by)
            l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval()
    with torch.no_grad(): preds=m(X_te.to(DEVICE)).argmax(dim=1).cpu()
    return confusion_matrix(y_te,preds),accuracy_score(y_te,preds)

# ═══════════════════════════════════════════════════
data_configs = []

# UNSW-NB15
def load_unsw(multiclass=False):
    tr=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
    te=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
    df=pd.concat([tr,te],ignore_index=True)
    df=df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns],errors='ignore')
    if multiclass:
        yr=df['attack_cat'].astype(str).str.strip().replace('','Normal')
        y=LabelEncoder().fit_transform(yr); cls=list(LabelEncoder().fit(yr).classes_)
    else: y=df['label'].values.astype(np.int64); cls=['Normal','Attack']
    feats=[c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
    X=df[feats]
    for c in X.select_dtypes(include=['object']).columns: X[c]=LabelEncoder().fit_transform(X[c].astype(str))
    X=X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X=StandardScaler().fit_transform(X)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    return torch.from_numpy(X_tr).float(),torch.from_numpy(y_tr).long(),torch.from_numpy(X_te).float(),torch.from_numpy(y_te).long(),cls

for task, mc in [('binary',False),('multiclass',True)]:
    X_tr,y_tr,X_te,y_te,cls=load_unsw(multiclass=mc)
    data_configs.append(('UNSW-NB15',task,X_tr,y_tr,X_te,y_te,cls))

# CIC-IDS-2017
def load_cic_data(path, max_s=60000):
    d=load_cicflowmeter(path,rows_per_file=20000,max_total=max_s,binary=True)
    X=torch.cat([d['X_train'][:40000],d['X_val'][:5000],d['X_test'][:15000]],dim=0)
    y=torch.cat([d['y_train'][:40000],d['y_val'][:5000],d['y_test'][:15000]],dim=0)
    n=int(len(X)*0.7)
    return X[:n],y[:n],X[n:],y[n:],['Normal','Attack']

X_tr,y_tr,X_te,y_te,cls=load_cic_data('/opt/CIC-IDS-2017/MachineLearningCVE',60000)
data_configs.append(('CIC-IDS-2017','binary',X_tr,y_tr,X_te,y_te,cls))

# CIC-DDoS2019
X_tr,y_tr,X_te,y_te,cls=load_cic_data('/opt/CIC-DDoS2019/all',60000)
data_configs.append(('CIC-DDoS2019','binary',X_tr,y_tr,X_te,y_te,cls))

# NSL-KDD
COL=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
     'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
     'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
     'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
     'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
     'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
     'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']

for task, mc in [('binary',False),('multiclass',True)]:
    df_tr=pd.read_csv('/opt/NSL-KDD/KDDTrain+.txt',names=COL)
    df_te=pd.read_csv('/opt/NSL-KDD/KDDTest+.txt',names=COL)
    df=pd.concat([df_tr,df_te],ignore_index=True)
    for c in ['protocol_type','service','flag']:
        df[c]=LabelEncoder().fit_transform(df[c].astype(str))
    if mc:
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
    X_tr,y_tr,X_te,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    data_configs.append(('NSL-KDD',task,torch.from_numpy(X_tr).float(),torch.from_numpy(y_tr).long(),
                         torch.from_numpy(X_te).float(),torch.from_numpy(y_te).long(),cls))

# ═══════ RUN ALL ═══════
print(f'Total configs: {len(data_configs)}')
t0=time.time()

for ds,task,X_tr,y_tr,X_te,y_te,cls in data_configs:
    nc=len(cls)
    print(f'\n{ds} {task} ({nc} classes, train={len(X_tr)}, test={len(X_te)})')
    
    # FeatureTransformer
    cm_ft,acc_ft=train_feattrans(X_tr,y_tr,X_te,y_te,nc)
    plot_cm(cm_ft,cls,f'{ds} - FeatureTransformer ({task})',os.path.join(OUT,f'cm_{ds}_{task}_feattrans.png'))
    print(f'  FeatTrans: Acc={acc_ft:.4f}')
    
    # GraphSAGE
    cm_gnn,acc_gnn=train_gnn(X_tr,y_tr,X_te,y_te,nc)
    plot_cm(cm_gnn,cls,f'{ds} - GraphSAGE ({task})',os.path.join(OUT,f'cm_{ds}_{task}_gnn.png'))
    print(f'  GNN: Acc={acc_gnn:.4f}')
    
    # Full Model
    cm_full,acc_full=train_full(X_tr,y_tr,X_te,y_te,nc)
    plot_cm(cm_full,cls,f'{ds} - Our Full Model ({task})',os.path.join(OUT,f'cm_{ds}_{task}_full.png'))
    print(f'  Full: Acc={acc_full:.4f}')
    
    # Side-by-side comparison
    fig,axes=plt.subplots(1,3,figsize=(20,6) if nc<=5 else (20,7))
    fig.suptitle(f'{ds} ({task}, {nc} classes)',fontsize=13,fontweight='bold')
    for i,(cm_data,name) in enumerate(zip([cm_ft,cm_gnn,cm_full],['FeatureTransformer','GraphSAGE','Our Full Model'])):
        ax=axes[i]
        im=ax.imshow(cm_data,cmap='Blues')
        n=len(cls)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(cls,rotation=45,ha='right',fontsize=6)
        ax.set_yticklabels(cls,fontsize=6)
        ax.set_xlabel('Pred'); ax.set_title(name,fontsize=9,fontweight='bold')
        plt.colorbar(im,ax=ax,shrink=0.7)
    plt.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(os.path.join(OUT,f'cm_{ds}_{task}_comparison.png'),dpi=200,bbox_inches='tight'); plt.close()
    print(f'  Comparison saved')

print(f'\nDone. {len(os.listdir(OUT))} files. Time: {(time.time()-t0)/60:.0f}min')
