# -*- coding: utf-8 -*-
"""Multi-seed stabilisation of the DCCA-honest comparison.
For 5 seeds x 16 subjects, compute LogReg-honest and DCCA-honest (trial-grouped);
average per subject across seeds, then paired tests on the stabilised per-subject
values. Reports across-seed mean+/-std so the margin is not a single noisy draw.
Leaky DCCA = 100.0 (established across all prior runs: dcca_both, validate, reconcile)."""
import os, pickle, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "results", "workstation"); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("SEEDV_DATA", os.path.join(_REPO, "data"))
import torch, torch.nn as nn
import scipy.stats as st
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression

def load(sub):
    de=np.load(os.path.join(DATA,"EEG_DE_features",f"{sub}_123.npz"),allow_pickle=True)
    ey=np.load(os.path.join(DATA,"Eye_movement_features",f"{sub}_123.npz"),allow_pickle=True)
    dd=pickle.loads(de["data"].tobytes()); dl=pickle.loads(de["label"].tobytes()); ed=pickle.loads(ey["data"].tobytes())
    Xe,Xy,Y,G,S=[],[],[],[],[]
    for t in sorted(dd.keys()):
        e=np.asarray(dd[t],np.float32); y=np.asarray(ed[t],np.float32); lab=np.asarray(dl[t]).astype(int).ravel()
        n=min(e.shape[0],y.shape[0],lab.shape[0])
        Xe.append(e[:n]);Xy.append(y[:n]);Y.append(lab[:n]);G.append(np.full(n,t));S.append(np.full(n,t//15))
    return np.concatenate(Xe),np.concatenate(Xy),np.concatenate(Y),np.concatenate(G),np.concatenate(S)
def snorm(X,S):
    X=X.copy()
    for s in np.unique(S): m=S==s; X[m]=(X[m]-np.nanmean(X[m],0))/(np.nanstd(X[m],0)+1e-6)
    return np.nan_to_num(X).astype(np.float32)
def cca_loss(H1,H2,r=1e-3,eps=1e-9):
    H1,H2=H1.t(),H2.t(); m=H1.shape[1]; d=H1.shape[0]
    H1=H1-H1.mean(1,keepdim=True); H2=H2-H2.mean(1,keepdim=True)
    S12=(H1@H2.t())/(m-1); S11=(H1@H1.t())/(m-1)+r*torch.eye(d); S22=(H2@H2.t())/(m-1)+r*torch.eye(d)
    D1,V1=torch.linalg.eigh(S11); D2,V2=torch.linalg.eigh(S22); D1=torch.clamp(D1,eps); D2=torch.clamp(D2,eps)
    T=(V1@torch.diag(D1**-0.5)@V1.t())@S12@(V2@torch.diag(D2**-0.5)@V2.t())
    return -torch.clamp(torch.linalg.svdvals(T),0,1).sum()
class Net(nn.Module):
    def __init__(s,din,dout=32): super().__init__(); s.f=nn.Sequential(nn.Linear(din,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,dout))
    def forward(s,x): return s.f(x)
def dcca_acc(Xe,Xy,Y,tr,te,ep=120):
    fe=Net(Xe.shape[1]); fy=Net(Xy.shape[1]); opt=torch.optim.Adam(list(fe.parameters())+list(fy.parameters()),1e-3,weight_decay=1e-4)
    xe,xy=torch.tensor(Xe[tr]),torch.tensor(Xy[tr])
    for _ in range(ep): opt.zero_grad(); cca_loss(fe(xe),fy(xy)).backward(); opt.step()
    fe.eval(); fy.eval()
    with torch.no_grad():
        Ztr=np.concatenate([fe(xe).numpy(),fy(xy).numpy()],1); Zte=np.concatenate([fe(torch.tensor(Xe[te])).numpy(),fy(torch.tensor(Xy[te])).numpy()],1)
    return (LogisticRegression(max_iter=500).fit(Ztr,Y[tr]).predict(Zte)==Y[te]).mean()
def lr_acc(Xc,Y,tr,te):
    return (LogisticRegression(max_iter=500).fit(Xc[tr],Y[tr]).predict(Xc[te])==Y[te]).mean()

DAT={s:(lambda a:(snorm(a[0],a[4]),snorm(a[1],a[4]),a[2],a[3]))(load(s)) for s in range(1,17)}
SEEDS=[0,1,2,3,4]; t0=time.time()
lr_by_seed=np.zeros((len(SEEDS),16)); dc_by_seed=np.zeros((len(SEEDS),16))
for si,seed in enumerate(SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    for j,s in enumerate(range(1,17)):
        Xe,Xy,Y,G=DAT[s]; Xc=np.concatenate([Xe,Xy],1)
        a=[]; d=[]
        for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(Xc,Y,G):
            a.append(lr_acc(Xc,Y,tr,te)); d.append(dcca_acc(Xe,Xy,Y,tr,te))
        lr_by_seed[si,j]=np.mean(a)*100; dc_by_seed[si,j]=np.mean(d)*100
    print(f"  seed {seed}: LogReg-honest {lr_by_seed[si].mean():5.2f} | DCCA-honest {dc_by_seed[si].mean():5.2f}  ({time.time()-t0:.0f}s)",flush=True)

lr_overall=lr_by_seed.mean(1); dc_overall=dc_by_seed.mean(1)      # per-seed overall means
lr_bar=lr_by_seed.mean(0); dc_bar=dc_by_seed.mean(0)              # per-subject, averaged over seeds
_,p_t=st.ttest_rel(lr_bar,dc_bar); _,p_w=st.wilcoxon(lr_bar,dc_bar)
win=int((lr_bar>dc_bar).sum())
L=[]
L.append("=== MULTI-SEED DCCA STABILISATION (5 seeds x 16 subjects) ===")
L.append(f"LogReg-honest: {lr_overall.mean():.2f} +/- {lr_overall.std():.2f} (across seeds)")
L.append(f"DCCA-honest  : {dc_overall.mean():.2f} +/- {dc_overall.std():.2f} (across seeds)")
L.append(f"Stabilised margin (per-subject avg over seeds): {lr_bar.mean()-dc_bar.mean():+.2f} pts")
L.append(f"  paired t p={p_t:.2e} | Wilcoxon p={p_w:.2e} | LogReg wins {win}/16 subjects")
L.append(f"DCCA leaky = 100.0 (established); collapse = {100-dc_overall.mean():.1f} pts")
txt="\n".join(L); print("\n"+txt,flush=True)
open(os.path.join(_OUT, "RESULTS_dcca_multiseed.md"),"w",encoding="utf-8").write(
    txt+"\n\nPer-seed overall (LogReg / DCCA honest):\n"+
    "\n".join(f"seed {SEEDS[i]}: {lr_overall[i]:.2f} / {dc_overall[i]:.2f}" for i in range(len(SEEDS)))+"\n")
print(f"\nsaved RESULTS_dcca_multiseed.md ({time.time()-t0:.0f}s)",flush=True)
