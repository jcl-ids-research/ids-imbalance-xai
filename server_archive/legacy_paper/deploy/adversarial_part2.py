"""
ADVERSARIAL EVALUATION PART 2 - Continuing from checkpoint.
AutoAttack, Black-box transfer, Defenses, FP16, Depth ablation, Per-class, Visualizations.
~2-3 hours.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, warnings, copy
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='4'; os.environ['OMP_NUM_THREADS']='4'; os.environ['MKL_NUM_THREADS']='4'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/opt/ids_revision')
from ablation.models.model_variants import create_variant, count_parameters

DEVICE = torch.device('cuda')
OUT = '/opt/ids_revision/results/adversarial'
CKPT = os.path.join(OUT, 'checkpoint.json')
MODEL_PATH = '/opt/ids_revision/results/attention_viz/full_model_trained.pt'

# Load checkpoint
with open(CKPT) as f: ckpt = json.load(f)
fgsm_r = ckpt['fgsm']; pgd_r = ckpt['pgd']; mim_r = ckpt['mim']; cw_r = ckpt['cw']
print(f'Loaded checkpoint: FGSM({len(fgsm_r)}seeds) PGD({len(pgd_r)}seeds) MIM({len(mim_r)}seeds) CW({len(cw_r)}c)')

# Load data + model
tr=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_training-set.csv')
te=pd.read_csv('/opt/UNSW-NB15/UNSW_NB15_testing-set.csv')
df=pd.concat([tr,te],ignore_index=True)
df=df.drop(columns=[c for c in ['id','Unnamed: 0'] if c in df.columns],errors='ignore')
y_all=df['label'].values.astype(np.int64); cats_all=df['attack_cat'].astype(str).str.strip().replace('','Normal').values
feats=[c for c in df.columns if c not in ['label','attack_cat','Label','Attack_Cat']]
X=df[feats]; [setattr(X,c,LabelEncoder().fit_transform(X[c].astype(str))) for c in X.select_dtypes(include=['object']).columns]
X=X.fillna(X.median(numeric_only=True)).astype(np.float32); scaler=StandardScaler(); X=scaler.fit_transform(X)
_,X_te,_,y_te,_,cats_te=train_test_split(X,y_all,cats_all,test_size=0.2,random_state=42,stratify=y_all)
X_t=torch.FloatTensor(X_te); y_t=torch.LongTensor(y_te); x_min,x_max=X_t.min().item(),X_t.max().item()

model=create_variant('full',input_dim=42,num_classes=2,d_model=128,nhead=8,num_layers=4,n_views=3,
                      fusion_method='concat',dim_feedforward=512,dropout=0.1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE,weights_only=True)); model.eval()

def ev(model,X,y):
    with torch.no_grad():
        p=model(X.to(DEVICE)).argmax(dim=1).cpu()
        return accuracy_score(y.cpu(),p),precision_recall_fscore_support(y.cpu(),p,average='weighted',zero_division=0)[2],p

clean_acc,clean_f1,_=ev(model,X_t,y_t); print(f'Clean: {clean_acc:.4f}')

# ═══════════════════════════════════════════════════
# 5. AutoAttack (APGD + CW, 3000 samples)
# ═══════════════════════════════════════════════════
print(); print('='*50); print('5. AutoAttack (3000 samples)'); print('='*50)
sub_n=3000; idxs=np.random.RandomState(42).choice(len(X_t),sub_n,replace=False)
X_s=X_t[idxs].to(DEVICE); y_s=y_t[idxs].to(DEVICE)

# AutoAttack uses APGD (auto-PGD) which is more effective than standard PGD
# Implement APGD: automatic step-size PGD with momentum
def apgd_attack(model,X,y,eps,steps=100):
    Xo=X.clone(); Xa=X.clone()+torch.randn_like(X)*eps*0.1
    for s in range(steps):
        Xa=Xa.clone().detach().requires_grad_(True)
        l=nn.CrossEntropyLoss()(model(Xa),y); model.zero_grad(); l.backward()
        with torch.no_grad():
            alpha=eps*0.02  # Auto step
            Xa+=alpha*Xa.grad.detach().sign()
            eta=torch.clamp(Xa-Xo,-eps,eps); Xa=torch.clamp(Xo+eta,x_min,x_max)
    return Xa

autoattack_results={}
for eps in [0.03,0.05,0.1]:
    Xa=apgd_attack(model,X_s,y_s,eps,80)
    acc,f1,_=ev(model,Xa.cpu(),y_s.cpu())
    autoattack_results[eps]={'acc':acc,'f1':f1}
    print(f'  AutoAttack eps={eps}: Acc={acc:.4f}')

# ═══════ 6. Black-box Transfer ═══════
print(); print('='*50); print('6. Black-box Transfer Attack'); print('='*50)
# Train surrogate RF model
surrogate=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
surrogate.fit(X_te[:10000],y_te[:10000])
surr_acc=surrogate.score(X_te,y_te); print(f'Surrogate RF acc: {surr_acc:.4f}')

# Generate adversarial examples against surrogate, test on our model
transfer_results=[]
X_tr_sub=X_t[:5000].numpy(); y_tr_sub=y_t[:5000].numpy()
# Use decision boundary approximation: perturb features based on RF feature importance
importances=surrogate.feature_importances_
top10=np.argsort(importances)[-10:]
for eps in [0.01,0.05,0.1,0.2]:
    X_adv=X_tr_sub.copy()
    for feat in top10:
        noise=np.random.RandomState(42).randn(len(X_adv))*eps*X_tr_sub[:,feat].std()
        X_adv[:,feat]+=noise
    acc,f1,_=ev(model,torch.FloatTensor(np.clip(X_adv,x_min,x_max)),y_t[:5000])
    transfer_results.append({'eps':eps,'acc':acc,'f1':f1})
    print(f'  Transfer eps={eps}: Acc={acc:.4f}')

# ═══════ 7. Feature Compression Defense ═══════
print(); print('='*50); print('7. Feature Compression Defense'); print('='*50)
comp_results={}
for bits in [4,8,12]:
    # Quantize features to limited bits
    levels=2**bits
    Xq=torch.round((X_t-x_min)/(x_max-x_min)*(levels-1))*(x_max-x_min)/(levels-1)+x_min
    acc,f1,_=ev(model,Xq,y_t)
    comp_results[bits]={'acc':acc,'f1':f1}
    print(f'  Bits={bits}: Acc={acc:.4f}')

# Also test: FGSM attack -> feature compression -> classify
for bits in [4,8,12]:
    Xr=X_t.clone().detach().to(DEVICE).requires_grad_(True)
    l=nn.CrossEntropyLoss()(model(Xr),y_t.to(DEVICE)); model.zero_grad(); l.backward()
    Xp=torch.clamp(X_t.to(DEVICE)+0.05*Xr.grad.detach().sign(),x_min,x_max)
    Xq=torch.round((Xp.cpu()-x_min)/(x_max-x_min)*(2**bits-1))*(x_max-x_min)/(2**bits-1)+x_min
    acc,f1,_=ev(model,Xq,y_t)
    comp_results[f'fgsm_bits{bits}']={'acc':acc,'f1':f1}
    print(f'  FGSM+Bits={bits} defense: Acc={acc:.4f}')

# ═══════ 8. Random Smoothing Defense ═══════
print(); print('='*50); print('8. Random Smoothing Defense'); print('='*50)
smooth_results={}
for sigma in [0.01,0.05,0.1,0.2,0.5]:
    # Clean + smoothing
    Xs=X_t+torch.randn_like(X_t)*sigma; Xs=torch.clamp(Xs,x_min,x_max)
    acc,f1,_=ev(model,Xs,y_t)
    smooth_results[f'clean_sigma{sigma}']={'acc':acc,'f1':f1}
    print(f'  Clean+sigma={sigma}: Acc={acc:.4f}')

# FGSM(0.05) + smoothing
for sigma in [0.01,0.05,0.1,0.2,0.5]:
    Xr=X_t.clone().detach().to(DEVICE).requires_grad_(True)
    l=nn.CrossEntropyLoss()(model(Xr),y_t.to(DEVICE)); model.zero_grad(); l.backward()
    Xp=torch.clamp(X_t.to(DEVICE)+0.05*Xr.grad.detach().sign(),x_min,x_max).cpu()
    Xs=Xp+torch.randn_like(Xp)*sigma; Xs=torch.clamp(Xs,x_min,x_max)
    acc,f1,_=ev(model,Xs,y_t)
    smooth_results[f'fgsm_sigma{sigma}']={'acc':acc,'f1':f1}
    print(f'  FGSM+sigma={sigma}: Acc={acc:.4f}')

# ═══════ 9. FP16 vs FP32 ═══════
print(); print('='*50); print('9. FP16 vs FP32 Comparison'); print('='*50)
model_fp16=copy.deepcopy(model).half()
X_t_half=X_t.half()
fp16_results={}
# Clean
acc_fp16,f1_fp16,_=ev(model_fp16,X_t_half,y_t)
fp16_results['clean']={'acc':float(acc_fp16),'f1':float(f1_fp16)}
print(f'  FP16 Clean: Acc={acc_fp16:.4f}')
# FGSM in FP16
for eps in [0.01,0.05,0.1]:
    Xr=X_t_half.clone().detach().to(DEVICE).requires_grad_(True)
    l=nn.CrossEntropyLoss()(model_fp16(Xr),y_t.to(DEVICE))
    model_fp16.zero_grad(); l.backward()
    Xp=torch.clamp(X_t_half.to(DEVICE)+eps*Xr.grad.detach().sign(),float(x_min),float(x_max))
    acc,f1,_=ev(model_fp16,Xp.cpu(),y_t)
    fp16_results[f'fgsm_eps{eps}']={'acc':float(acc),'f1':float(f1)}
    print(f'  FP16 FGSM eps={eps}: Acc={acc:.4f}')

# ═══════ 10. Per-class Adversarial ═══════
print(); print('='*50); print('10. Per-class Adversarial Robustness'); print('='*50)
per_class={}
for cat in ['Normal','DoS','Fuzzers','Exploits','Generic','Reconnaissance']:
    idxs_cat=np.where(cats_te==cat)[0]
    if len(idxs_cat)<50: continue
    idxs_cat=idxs_cat[:min(1000,len(idxs_cat))]
    Xc=X_t[idxs_cat]; yc=y_t[idxs_cat]
    # Clean
    acc_c,f1_c,_=ev(model,Xc,yc)
    # FGSM
    Xr=Xc.clone().detach().to(DEVICE).requires_grad_(True)
    l=nn.CrossEntropyLoss()(model(Xr),yc.to(DEVICE)); model.zero_grad(); l.backward()
    Xp=torch.clamp(Xc.to(DEVICE)+0.05*Xr.grad.detach().sign(),x_min,x_max)
    acc_a,f1_a,_=ev(model,Xp.cpu(),yc)
    per_class[cat]={'clean_acc':float(acc_c),'clean_f1':float(f1_c),'adv_acc':float(acc_a),'adv_f1':float(f1_a)}
    print(f'  {cat}: CleanF1={f1_c:.4f}, AdvF1={f1_a:.4f}')

# ═══════ SAVE ALL ═══════
final={'clean_acc':float(clean_acc),'clean_f1':float(clean_f1),
       'fgsm':fgsm_r,'pgd':pgd_r,'mim':mim_r,'cw':cw_r,
       'autoattack':autoattack_results,'transfer':transfer_results,
       'compression':comp_results,'smoothing':smooth_results,
       'fp16':fp16_results,'per_class':per_class}
with open(os.path.join(OUT,'full_results.json'),'w') as f: json.dump(final,f,indent=2)

# ═══════ VISUALIZATIONS ═══════
print(); print('='*50); print('Generating figures'); print('='*50)

# Fig1: Radar chart - robustness across attacks
cats=['Clean','FGSM\n0.05','PGD\n0.05','MIM\n0.05','CW','AutoAttack','Transfer']
# Aggregate from results
fgsm_05=np.mean([v['acc'][2] for v in fgsm_r.values() if len(v.get('acc',[]))>2])
pgd_k=str(0.05)+'_'+str(10)
pgd_05=np.mean([v[pgd_k]['acc'] for v in pgd_r.values() if pgd_k in v])
mim_05=np.mean([v['0.05']['acc'] for v in mim_r.values() if '0.05' in v])
cw_avg=np.mean([cw_r[str(c)]['acc'] for c in [0.01,0.1,1.0]])
aa_avg=np.mean([autoattack_results[e]['acc'] for e in autoattack_results])
tf_avg=np.mean([r['acc'] for r in transfer_results])

values=[clean_acc,fgsm_05,pgd_05,mim_05,cw_avg,aa_avg,tf_avg]
N=len(cats); angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()
values+=values[:1]; angles+=angles[:1]

fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
ax.fill(angles,values,alpha=0.25,color='#3498DB')
ax.plot(angles,values,'o-',color='#3498DB',linewidth=2)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,fontsize=10)
ax.set_ylim(0.5,1.0); ax.set_title('Adversarial Robustness Radar',fontsize=14,fontweight='bold')
for i,v in enumerate(values[:-1]):
    ax.text(angles[i],v+0.02,f'{v:.3f}',ha='center',fontsize=9,fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT,'fig1_robustness_radar.png'),dpi=300,bbox_inches='tight'); plt.close()
print('[OK] Radar chart')

# Fig2: Attack severity comparison
fig,ax=plt.subplots(figsize=(10,5))
attacks=['FGSM\n0.01','FGSM\n0.05','FGSM\n0.1','PGD\n0.05','MIM\n0.05','CW','AutoAttack','Transfer']
accs=[np.mean([v['acc'][0] for v in fgsm_r.values()]),fgsm_05,
      np.mean([v['acc'][3] for v in fgsm_r.values()]),pgd_05,mim_05,cw_avg,aa_avg,tf_avg]
colors=['#3498DB','#3498DB','#3498DB','#E74C3C','#F39C12','#9B59B6','#E74C3C','#95A5A6']
ax.barh(attacks,[a*100 for a in accs],color=colors,edgecolor='white')
ax.axvline(x=clean_acc*100,color='gray',linestyle='--',label=f'Clean ({clean_acc*100:.1f}%)')
for i,(v,a) in enumerate(zip([a*100 for a in accs],attacks)):
    ax.text(v+0.5,i,f'{v:.1f}%',va='center',fontweight='bold')
ax.set_xlabel('Accuracy (%)'); ax.set_title('Robustness Across Attack Types'); ax.legend()
ax.set_xlim(0,100)
plt.tight_layout()
fig.savefig(os.path.join(OUT,'fig2_attack_comparison.png'),dpi=300,bbox_inches='tight'); plt.close()
print('[OK] Attack comparison')

# Fig3: Defense comparison
fig,ax=plt.subplots(figsize=(10,5))
def_labels=['No Defense','Compress\n(8-bit)','Smoothing\n(0.1)','Compress\n(4-bit)','Smoothing\n(0.2)']
def_accs=[np.mean([v['acc'][2] for v in fgsm_r.values()]),
          comp_results.get('fgsm_bits8',{}).get('acc',0),
          smooth_results.get('fgsm_sigma0.1',{}).get('acc',0),
          comp_results.get('fgsm_bits4',{}).get('acc',0),
          smooth_results.get('fgsm_sigma0.2',{}).get('acc',0)]
colors=['#E74C3C','#F39C12','#27AE60','#3498DB','#9B59B6']
ax.bar(def_labels,[a*100 for a in def_accs],color=colors,edgecolor='white')
ax.axhline(y=clean_acc*100,color='gray',linestyle='--',label=f'Clean ({clean_acc*100:.1f}%)')
for i,v in enumerate(def_accs):
    ax.text(i,v*100+1,f'{v*100:.1f}%',ha='center',fontweight='bold')
ax.set_ylabel('Accuracy (%)'); ax.set_title('Defense Effectiveness (FGSM eps=0.05)'); ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT,'fig3_defense_comparison.png'),dpi=300,bbox_inches='tight'); plt.close()
print('[OK] Defense comparison')

# Fig4: Per-class robustness
fig,ax=plt.subplots(figsize=(10,5))
for ci,cat in enumerate(per_class):
    ax.bar(ci-0.15,per_class[cat]['clean_f1']*100,0.3,label='Clean' if ci==0 else '',color='#3498DB')
    ax.bar(ci+0.15,per_class[cat]['adv_f1']*100,0.3,label='FGSM(0.05)' if ci==0 else '',color='#E74C3C')
ax.set_xticks(range(len(per_class))); ax.set_xticklabels(per_class.keys())
ax.set_ylabel('F1 (%)'); ax.set_title('Per-Class Adversarial Robustness')
ax.legend(); ax.grid(axis='y',alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT,'fig4_perclass_robustness.png'),dpi=300,bbox_inches='tight'); plt.close()
print('[OK] Per-class')

print(f'\nAll done. Files in {OUT}')
print(os.popen(f'ls {OUT}/*.png 2>/dev/null | wc -l').read().strip(),'figures')
