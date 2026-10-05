# -*- coding: utf-8 -*-
"""DCCA vs LogReg under trial-independent INDUCTIVE normalisation (honest) and the leaky split.
Honest folds: StratifiedGroupKFold(3), per-session stats fit on training trials only.
Leaky folds: StratifiedKFold(5) (sample-level; ceiling regardless of normalisation)."""
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
        n=min(e.shape[0],y.shape[0],lab.shape[0]); Xe.append(e[:n]);Xy.append(y[:n]);Y.append(lab[:n]);G.append(np.full(n,t));S.append(np.full(n,t//15))
    return np.concatenate(Xe),np.concatenate(Xy),np.concatenate(Y),np.concatenate(G),np.concatenate(S)
DAT={s:load(s) for s in range(1,17)}
def norm(Xe,Xy,S,tm,mode):
    Xe=Xe.copy(); Xy=Xy.copy()
    for s in np.unique(S):
        m=S==s; src=m if mode=="trans" else (m & tm)
        if src.sum()<2: src=m
        for X in (Xe,Xy):
            mu=np.nanmean(X[src],0); sd=np.nanstd(X[src],0); X[m]=np.nan_to_num((X[m]-mu)/(sd+1e-6))
    return Xe.astype(np.float32),Xy.astype(np.float32)
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
lr_h,dc_h,dc_lk=[],[],[]; t0=time.time()
for s in range(1,17):
    Xe,Xy,Y,G,S=DAT[s]
    a=[];d=[]
    for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=0).split(Xe,Y,G):
        tm=np.zeros(len(Y),bool); tm[tr]=True; Xe_n,Xy_n=norm(Xe,Xy,S,tm,"induct"); Xc=np.concatenate([Xe_n,Xy_n],1)
        a.append((LogisticRegression(max_iter=500).fit(Xc[tr],Y[tr]).predict(Xc[te])==Y[te]).mean())
        d.append(dcca_acc(Xe_n,Xy_n,Y,tr,te))
    dl=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xe,Y):
        tm=np.zeros(len(Y),bool); tm[tr]=True; Xe_n,Xy_n=norm(Xe,Xy,S,tm,"trans"); dl.append(dcca_acc(Xe_n,Xy_n,Y,tr,te))
    lr_h.append(np.mean(a)*100); dc_h.append(np.mean(d)*100); dc_lk.append(np.mean(dl)*100)
    print(f"  S{s:02d}: LRh {lr_h[-1]:.1f} DCh {dc_h[-1]:.1f} DClk {dc_lk[-1]:.1f} ({time.time()-t0:.0f}s)",flush=True)
lr_h,dc_h,dc_lk=map(np.array,(lr_h,dc_h,dc_lk))
_,p=st.ttest_rel(lr_h,dc_h); _,pw=st.wilcoxon(lr_h,dc_h)
out=(f"=== DCCA vs LogReg, trial-independent INDUCTIVE (16 subjects) ===\n"
 f"LogReg-honest(inductive): {lr_h.mean():.2f} +/- {lr_h.std():.1f}\n"
 f"DCCA-honest(inductive)  : {dc_h.mean():.2f} +/- {dc_h.std():.1f}\n"
 f"DCCA-leaky              : {dc_lk.mean():.2f} +/- {dc_lk.std():.1f}\n"
 f"LogReg - DCCA (honest inductive): {lr_h.mean()-dc_h.mean():+.2f} pts | t p={p:.2e} | Wilcoxon p={pw:.2e} | LR wins {int((lr_h>dc_h).sum())}/16\n"
 f"DCCA collapse leaky->honest: {dc_lk.mean()-dc_h.mean():+.2f} pts")
print("\n"+out,flush=True); open(os.path.join(_OUT, "RESULTS_dcca_inductive.md"),"w").write(out+"\n")
print(f"saved ({time.time()-t0:.0f}s)",flush=True)
