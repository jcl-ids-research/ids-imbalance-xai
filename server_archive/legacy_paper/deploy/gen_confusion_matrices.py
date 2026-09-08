"""
Generate confusion matrices for FeatureTransformer and GraphSAGE.
Retrain both architectures on UNSW-NB15 (binary + multi-class).
~10-15 minutes on A100.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'; os.environ['OMP_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import torch.nn.functional as F

sys.path.insert(0, '/opt/ids_revision')
from ablation.models.feature_transformer import FeatureTransformer
from torch_geometric.nn import SAGEConv

DEVICE = torch.device('cuda')
OUT = '/opt/ids_revision/results/confusion_matrices'
os.makedirs(OUT, exist_ok=True)

def load_unsw(multiclass=False):
    tr=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
    te=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
    df=pd.concat([tr,te],ignore_index=True)
    df=df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns],errors='ignore')
    if multiclass:
        y_raw=df['attack_cat'].astype(str).str.strip().replace('','Normal')
        le=LabelEncoder(); y=le.fit_transform(y_raw); class_names=list(le.classes_)
    else:
        y=df['label'].values.astype(np.int64); class_names=['Normal','Attack']
    feats=[c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
    X=df[feats]
    for c in X.select_dtypes(include=['object']).columns: X[c]=LabelEncoder().fit_transform(X[c].astype(str))
    X=X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X=StandardScaler().fit_transform(X)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    print(f'  X_tr: {X_tr.shape}, y_tr: {y_tr.shape}, X_te: {X_te.shape}, y_te: {y_te.shape}')
    return torch.from_numpy(X_tr).float(),torch.from_numpy(y_tr).long(),torch.from_numpy(X_te).float(),torch.from_numpy(y_te).long(),class_names

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
    fig,ax=plt.subplots(figsize=(8,7))
    im=ax.imshow(cm,cmap='Blues')
    n=len(class_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names,rotation=45,ha='right',fontsize=7)
    ax.set_yticklabels(class_names,fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title,fontsize=11,fontweight='bold')
    for i in range(n):
        for j in range(n):
            if cm[i,j]>0:
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6,
                       color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.colorbar(im,ax=ax,shrink=0.8)
    plt.tight_layout()
    fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close()

print('='*60)
print('CONFUSION MATRICES: FeatureTransformer + GraphSAGE')
print('='*60)

# ═══ 1. FeatureTransformer Binary ═══
print(); print('--- FeatureTransformer Binary ---')
X_tr,y_tr,X_te,y_te,cls=load_unsw(multiclass=False)
m=FeatureTransformer(n_features=42,num_classes=2,d_model=64,nhead=8,num_layers=4,pooling='mean').to(DEVICE)
opt=torch.optim.AdamW(m.parameters(),lr=0.0001)
n_samples=len(X_tr)
for ep in range(1,21):
    m.train()
    perm=torch.randperm(n_samples)
    for i in range(0,n_samples,256):
        idx=perm[i:i+256]
        bx,by=X_tr[idx].to(DEVICE),y_tr[idx].to(DEVICE)
        opt.zero_grad(); l=nn.CrossEntropyLoss()(m(bx),by)
        l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if ep%10==0: print(f'  Epoch {ep}/20')
m.eval()
with torch.no_grad():
    preds=m(X_te.to(DEVICE)).argmax(dim=1).cpu()
cm_ft_bin=confusion_matrix(y_te,preds)
acc=accuracy_score(y_te,preds)
print(f'  Acc={acc:.4f}')
plot_cm(cm_ft_bin,cls,'FeatureTransformer Binary',os.path.join(OUT,'cm_feattrans_binary.png'))
print('  Saved cm_feattrans_binary.png')

# ═══ 2. FeatureTransformer Multi-Class ═══
print(); print('--- FeatureTransformer Multi-Class ---')
X_tr_mc,y_tr_mc,X_te_mc,y_te_mc,cls_mc=load_unsw(multiclass=True)
nc=len(cls_mc)
m_mc=FeatureTransformer(n_features=42,num_classes=nc,d_model=64,nhead=8,num_layers=4,pooling='mean').to(DEVICE)
dl_mc=DataLoader(TensorDataset(X_tr_mc,y_tr_mc),batch_size=256,shuffle=True)
opt_mc=torch.optim.AdamW(m_mc.parameters(),lr=0.0001)
for ep in range(1,21):
    m_mc.train()
    for bx,by in dl_mc:
        opt_mc.zero_grad(); l=nn.CrossEntropyLoss()(m_mc(bx.to(DEVICE)),by.to(DEVICE))
        l.backward(); nn.utils.clip_grad_norm_(m_mc.parameters(),1.0); opt_mc.step()
m_mc.eval()
with torch.no_grad():
    preds_mc=m_mc(X_te_mc.to(DEVICE)).argmax(dim=1).cpu()
cm_ft_mc=confusion_matrix(y_te_mc,preds_mc)
acc_mc=accuracy_score(y_te_mc,preds_mc)
print(f'  Acc={acc_mc:.4f}')
plot_cm(cm_ft_mc,cls_mc,'FeatureTransformer Multi-Class (10 categories)',os.path.join(OUT,'cm_feattrans_multiclass.png'))
print('  Saved cm_feattrans_multiclass.png')

# ═══ 3. GraphSAGE Binary ═══
print(); print('--- GraphSAGE Binary ---')
X_tr_g,y_tr_g,X_te_g,y_te_g,cls_g=load_unsw(multiclass=False)
X_all=torch.cat([X_tr_g[:30000],X_te_g[:20000]],dim=0)
y_all=torch.cat([y_tr_g[:30000],y_te_g[:20000]],dim=0)
adj=kneighbors_graph(X_all.numpy(),n_neighbors=5,mode='connectivity',include_self=False)
ei=torch.tensor(np.array(adj.nonzero()),dtype=torch.long)
m_g=GraphSAGE(42,128,2).to(DEVICE)
opt_g=torch.optim.Adam(m_g.parameters(),lr=0.001,weight_decay=5e-4)
for ep in range(1,51):
    m_g.train(); opt_g.zero_grad()
    l=nn.CrossEntropyLoss()(m_g(X_all.to(DEVICE),ei.to(DEVICE))[:30000],y_all[:30000].to(DEVICE))
    l.backward(); nn.utils.clip_grad_norm_(m_g.parameters(),1.0); opt_g.step()
m_g.eval()
with torch.no_grad():
    preds_g=m_g(X_all.to(DEVICE),ei.to(DEVICE))[30000:].argmax(dim=1).cpu()
cm_g_bin=confusion_matrix(y_all[30000:],preds_g)
acc_g=accuracy_score(y_all[30000:],preds_g)
print(f'  Acc={acc_g:.4f}')
plot_cm(cm_g_bin,cls_g,'GraphSAGE Binary',os.path.join(OUT,'cm_gnn_binary.png'))
print('  Saved cm_gnn_binary.png')

# ═══ 4. GraphSAGE Multi-Class ═══
print(); print('--- GraphSAGE Multi-Class ---')
X_tr_gm,y_tr_gm,X_te_gm,y_te_gm,cls_gm=load_unsw(multiclass=True)
nc_g=len(cls_gm)
X_all_m=torch.cat([X_tr_gm[:30000],X_te_gm[:20000]],dim=0)
y_all_m=torch.cat([y_tr_gm[:30000],y_te_gm[:20000]],dim=0)
adj_m=kneighbors_graph(X_all_m.numpy(),n_neighbors=5,mode='connectivity',include_self=False)
ei_m=torch.tensor(np.array(adj_m.nonzero()),dtype=torch.long)
m_gm=GraphSAGE(42,128,nc_g).to(DEVICE)
opt_gm=torch.optim.Adam(m_gm.parameters(),lr=0.001,weight_decay=5e-4)
for ep in range(1,51):
    m_gm.train(); opt_gm.zero_grad()
    l=nn.CrossEntropyLoss()(m_gm(X_all_m.to(DEVICE),ei_m.to(DEVICE))[:30000],y_all_m[:30000].to(DEVICE))
    l.backward(); nn.utils.clip_grad_norm_(m_gm.parameters(),1.0); opt_gm.step()
m_gm.eval()
with torch.no_grad():
    preds_gm=m_gm(X_all_m.to(DEVICE),ei_m.to(DEVICE))[30000:].argmax(dim=1).cpu()
cm_g_mc=confusion_matrix(y_all_m[30000:],preds_gm)
acc_gm=accuracy_score(y_all_m[30000:],preds_gm)
print(f'  Acc={acc_gm:.4f}')
plot_cm(cm_g_mc,cls_gm,'GraphSAGE Multi-Class (10 categories)',os.path.join(OUT,'cm_gnn_multiclass.png'))
print('  Saved cm_gnn_multiclass.png')

# ═══ 5. Side-by-side comparison (multi-class) ═══
print(); print('--- Side-by-side comparison ---')
fig,axes=plt.subplots(1,3,figsize=(20,6))
fig.suptitle('Confusion Matrix Comparison: Multi-Class UNSW-NB15',fontsize=14,fontweight='bold')

models_cm=[(cm_ft_mc,'FeatureTransformer'),(cm_g_mc,'GraphSAGE'),(np.array(json.load(open('/opt/ids_revision/results/full_results.json'))['confusion_matrix']),'Our Full Model')]
for i,(cm_data,name) in enumerate(models_cm):
    ax=axes[i]
    im=ax.imshow(cm_data,cmap='Blues')
    n=len(cls_mc)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cls_mc,rotation=45,ha='right',fontsize=6)
    ax.set_yticklabels(cls_mc,fontsize=6)
    ax.set_title(name,fontsize=10,fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.colorbar(im,ax=ax,shrink=0.7)
plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(os.path.join(OUT,'cm_comparison_multiclass.png'),dpi=200,bbox_inches='tight'); plt.close()
print('  Saved cm_comparison_multiclass.png')

print(f'\nDone. {len(os.listdir(OUT))} files in {OUT}')
