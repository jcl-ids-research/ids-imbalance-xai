"""Quick FeatTrans CIC-IDS-2017 metrics."""
import os,sys,numpy as np,torch,torch.nn as nn,warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

sys.path.insert(0,'/opt/ids_revision'); sys.path.insert(0,'/opt/ids_revision/deploy')
from ablation.models.feature_transformer import FeatureTransformer
from cic_data_loader import load_cicflowmeter

DEVICE=torch.device('cuda')
data=load_cicflowmeter('/opt/CIC-IDS-2017/MachineLearningCVE',rows_per_file=20000,max_total=60000,binary=True)
X=torch.cat([data['X_train'][:40000],data['X_val'][:5000],data['X_test'][:15000]],dim=0)
y=torch.cat([data['y_train'][:40000],data['y_val'][:5000],data['y_test'][:15000]],dim=0)
n=int(len(X)*0.7); X_tr,y_tr,X_te,y_te=X[:n],y[:n],X[n:],y[n:]
print(f'train={len(X_tr)} test={len(X_te)}')

m=FeatureTransformer(n_features=76,num_classes=2,d_model=64,nhead=8,num_layers=4,pooling='mean').to(DEVICE)
opt=torch.optim.AdamW(m.parameters(),lr=0.0001)
for ep in range(1,16):
    m.train(); perm=torch.randperm(len(X_tr))
    for i in range(0,len(X_tr),256):
        idx=perm[i:i+256]; bx,by=X_tr[idx].to(DEVICE),y_tr[idx].to(DEVICE)
        opt.zero_grad(); l=nn.CrossEntropyLoss()(m(bx),by)
        l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
m.eval()
with torch.no_grad(): preds=m(X_te.to(DEVICE)).argmax(dim=1).cpu()
acc=accuracy_score(y_te,preds)
p,r,f1,_=precision_recall_fscore_support(y_te,preds,average='weighted',zero_division=0)
import json
print(json.dumps({'acc':round(float(acc),4),'prec':round(float(p),4),'rec':round(float(r),4),'f1':round(float(f1),4)}))
