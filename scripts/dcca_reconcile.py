# -*- coding: utf-8 -*-
"""ONE-RUN reconciliation of all DCCA-related numbers so the paper is internally consistent.
Computes, per subject and from the SAME RNG stream: LogReg & DCCA under leaky and honest
protocols, then every derived mean, margin, and paired test. Replaces the previously
mismatched dcca_both (DCCA-honest 69.94) vs validate (DCCA-honest ~68.0) values."""
import os, pickle, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "results", "workstation"); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("SEEDV_DATA", os.path.join(_REPO, "data"))
import torch, torch.nn as nn
import scipy.stats as st
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
torch.manual_seed(0); np.random.seed(0)

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
lr_h,dc_h,lr_lk,dc_lk=[],[],[],[]; t0=time.time()
for s in range(1,17):
    Xe,Xy,Y,G=DAT[s]; Xc=np.concatenate([Xe,Xy],1)
    # honest folds (trial-grouped)
    a=[]; d=[]
    for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=0).split(Xc,Y,G):
        a.append(lr_acc(Xc,Y,tr,te)); d.append(dcca_acc(Xe,Xy,Y,tr,te))
    # leaky folds (sample-level)
    al=[]; dl=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xc,Y):
        al.append(lr_acc(Xc,Y,tr,te)); dl.append(dcca_acc(Xe,Xy,Y,tr,te))
    lr_h.append(np.mean(a)*100); dc_h.append(np.mean(d)*100); lr_lk.append(np.mean(al)*100); dc_lk.append(np.mean(dl)*100)
    print(f"  S{s:02d}: LRh {lr_h[-1]:5.1f} DCh {dc_h[-1]:5.1f} | LRlk {lr_lk[-1]:5.1f} DClk {dc_lk[-1]:5.1f}  ({time.time()-t0:.0f}s)",flush=True)

lr_h,dc_h,lr_lk,dc_lk=map(np.array,(lr_h,dc_h,lr_lk,dc_lk))
# paired tests
_,p_lr_dc=st.ttest_rel(lr_h,dc_h); _,pw_lr_dc=st.wilcoxon(lr_h,dc_h)
_,p_lr_leak=st.ttest_rel(lr_lk,lr_h); _,pw_lr_leak=st.wilcoxon(lr_lk,lr_h)
_,p_dc_leak=st.ttest_rel(dc_lk,dc_h)
win=int((lr_h>dc_h).sum())
L=[]
L.append("=== DCCA RECONCILED (single run, 16 subjects) ===")
L.append(f"LogReg-honest : {lr_h.mean():.2f} +/- {lr_h.std():.1f}")
L.append(f"DCCA-honest   : {dc_h.mean():.2f} +/- {dc_h.std():.1f}")
L.append(f"LogReg-leaky  : {lr_lk.mean():.2f} +/- {lr_lk.std():.1f}")
L.append(f"DCCA-leaky    : {dc_lk.mean():.2f} +/- {dc_lk.std():.1f}")
L.append(f"[A] LogReg-honest vs DCCA-honest: diff {lr_h.mean()-dc_h.mean():+.2f} pts | t p={p_lr_dc:.2e} | Wilcoxon p={pw_lr_dc:.2e} | LogReg wins {win}/16")
L.append(f"[B] DCCA leaky vs honest (collapse): diff {dc_lk.mean()-dc_h.mean():+.2f} pts | t p={p_dc_leak:.2e}")
L.append(f"[C] LogReg leaky vs honest: diff {lr_lk.mean()-lr_h.mean():+.2f} pts | t p={p_lr_leak:.2e} | Wilcoxon p={pw_lr_leak:.2e}")
txt="\n".join(L); print("\n"+txt,flush=True)
open(os.path.join(_OUT, "RESULTS_dcca_reconciled.md"),"w",encoding="utf-8").write(
    txt+"\n\nPer-subject (LRh/DCh/LRlk/DClk):\n"+
    "\n".join(f"S{i+1:02d}: {lr_h[i]:.1f}/{dc_h[i]:.1f}/{lr_lk[i]:.1f}/{dc_lk[i]:.1f}" for i in range(16))+"\n")
print(f"\nsaved RESULTS_dcca_reconciled.md ({time.time()-t0:.0f}s)",flush=True)
