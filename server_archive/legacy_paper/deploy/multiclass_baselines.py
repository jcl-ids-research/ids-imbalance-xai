"""
Multi-class baselines: RF + XGBoost + CNN + MLP on UNSW-NB15 (10-class) and NSL-KDD (5-class).
Full datasets, no shortcuts.
"""
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch, torch.nn as nn

OUT = '/opt/ids_revision/results/traditional_ml'
DEVICE = torch.device('cuda')

class CNN1D(nn.Module):
    def __init__(self, in_f, n_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,64,3,padding=1),nn.ReLU(),nn.BatchNorm1d(64),
            nn.Conv1d(64,128,3,padding=1),nn.ReLU(),nn.BatchNorm1d(128),
            nn.Conv1d(128,256,3,padding=1),nn.ReLU(),nn.BatchNorm1d(256))
        self.gap=nn.AdaptiveAvgPool1d(1); self.cls=nn.Linear(256,n_c)
    def forward(self,x): x=x.unsqueeze(1); x=self.conv(x); x=self.gap(x).squeeze(-1); return self.cls(x)

class MLP(nn.Module):
    def __init__(self,in_f,n_c):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_f,256),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(64,n_c))
    def forward(self,x): return self.net(x)

def train_torch(model,X_tr,y_tr,X_te,y_te,epochs=30):
    m=model.to(DEVICE); opt=torch.optim.Adam(m.parameters(),lr=0.001)
    n=len(X_tr)
    for ep in range(epochs):
        m.train(); perm=torch.randperm(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]; bx,by=X_tr[idx].to(DEVICE),y_tr[idx].to(DEVICE)
            opt.zero_grad(); loss=nn.CrossEntropyLoss()(m(bx),by); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): preds=m(X_te.to(DEVICE)).argmax(dim=1).cpu().numpy()
    return preds

# ═══ UNSW multi-class ═══
def load_unsw_mc():
    tr=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
    te=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
    df=pd.concat([tr,te],ignore_index=True)
    df=df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns],errors='ignore')
    yr=df['attack_cat'].astype(str).str.strip().replace('','Normal')
    y=LabelEncoder().fit_transform(yr); cls=list(LabelEncoder().fit(yr).classes_)
    feats=[c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
    X=df[feats]
    for c in X.select_dtypes(include=['object']).columns: X[c]=LabelEncoder().fit_transform(X[c].astype(str))
    X=X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X=StandardScaler().fit_transform(X)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    return torch.FloatTensor(X_tr),torch.LongTensor(y_tr),torch.FloatTensor(X_te),torch.LongTensor(y_te),cls

# ═══ NSL multi-class ═══
COL=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
     'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
     'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
     'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
     'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
     'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
     'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']

def load_nsl_mc():
    df_tr=pd.read_csv('/opt/NSL-KDD/KDDTrain+.txt',names=COL)
    df_te=pd.read_csv('/opt/NSL-KDD/KDDTest+.txt',names=COL)
    df=pd.concat([df_tr,df_te],ignore_index=True)
    for c in ['protocol_type','service','flag']: df[c]=LabelEncoder().fit_transform(df[c].astype(str))
    attack_map={'normal':'normal','back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS','teardrop':'DoS',
                'satan':'Probe','ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
                'guess_passwd':'R2L','ftp_write':'R2L','imap':'R2L','phf':'R2L','multihop':'R2L','warezmaster':'R2L','warezclient':'R2L','spy':'R2L',
                'buffer_overflow':'U2R','loadmodule':'U2R','rootkit':'U2R','perl':'U2R'}
    yr=df['label'].map(attack_map).fillna('normal')
    y=LabelEncoder().fit_transform(yr); cls=list(LabelEncoder().fit(yr).classes_)
    feats=[c for c in COL if c not in ['label','difficulty']]
    X=df[feats].fillna(0).astype(np.float32)
    X=StandardScaler().fit_transform(X)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    return torch.FloatTensor(X_tr),torch.LongTensor(y_tr),torch.FloatTensor(X_te),torch.LongTensor(y_te),cls

# ═══ RUN ═══
all_mc={}
configs=[('UNSW-NB15',load_unsw_mc),('NSL-KDD',load_nsl_mc)]
t0=time.time()

for ds,loader in configs:
    X_tr,y_tr,X_te,y_te,cls=loader()
    nc=len(cls)
    print(f'\n{ds}: train={X_tr.shape}, test={X_te.shape}, classes={nc} ({cls})')
    all_mc[ds]={}
    
    # RF
    t1=time.time(); rf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
    rf.fit(X_tr.numpy(),y_tr.numpy()); preds=rf.predict(X_te.numpy())
    acc=accuracy_score(y_te,preds); p,r,f1,_=precision_recall_fscore_support(y_te,preds,average='weighted',zero_division=0)
    all_mc[ds]['RF']={'acc':float(acc),'prec':float(p),'rec':float(r),'f1':float(f1),'time':time.time()-t1}
    print(f'  RF: Acc={acc:.4f}, F1={f1:.4f}, Time={all_mc[ds]["RF"]["time"]:.1f}s')
    
    # XGBoost
    t1=time.time(); xm=xgb.XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=6,random_state=42,use_label_encoder=False,eval_metric='mlogloss')
    xm.fit(X_tr.numpy(),y_tr.numpy()); preds=xm.predict(X_te.numpy())
    acc=accuracy_score(y_te,preds); p,r,f1,_=precision_recall_fscore_support(y_te,preds,average='weighted',zero_division=0)
    all_mc[ds]['XGBoost']={'acc':float(acc),'prec':float(p),'rec':float(r),'f1':float(f1),'time':time.time()-t1}
    print(f'  XGB: Acc={acc:.4f}, F1={f1:.4f}, Time={all_mc[ds]["XGBoost"]["time"]:.1f}s')
    
    # CNN
    t1=time.time(); preds=train_torch(CNN1D(X_tr.shape[1],nc),X_tr,y_tr,X_te,y_te,30)
    acc=accuracy_score(y_te,preds); p,r,f1,_=precision_recall_fscore_support(y_te,preds,average='weighted',zero_division=0)
    all_mc[ds]['CNN']={'acc':float(acc),'prec':float(p),'rec':float(r),'f1':float(f1),'time':time.time()-t1}
    print(f'  CNN: Acc={acc:.4f}, F1={f1:.4f}, Time={all_mc[ds]["CNN"]["time"]:.1f}s')
    
    # MLP
    t1=time.time(); preds=train_torch(MLP(X_tr.shape[1],nc),X_tr,y_tr,X_te,y_te,30)
    acc=accuracy_score(y_te,preds); p,r,f1,_=precision_recall_fscore_support(y_te,preds,average='weighted',zero_division=0)
    all_mc[ds]['MLP']={'acc':float(acc),'prec':float(p),'rec':float(r),'f1':float(f1),'time':time.time()-t1}
    print(f'  MLP: Acc={acc:.4f}, F1={f1:.4f}, Time={all_mc[ds]["MLP"]["time"]:.1f}s')

import json
with open(os.path.join(OUT,'multiclass_baselines.json'),'w') as f: json.dump(all_mc,f,indent=2)
print(f'\nDone. Time: {(time.time()-t0)/60:.0f}min')
