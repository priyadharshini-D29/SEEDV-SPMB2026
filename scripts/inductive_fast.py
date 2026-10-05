# -*- coding: utf-8 -*-
"""Inductive (training-only normalisation) re-run of the fast LogReg experiments:
channel selection, 5-seed stability, and group-wise permutation importance.
Trial-independent StratifiedGroupKFold; per fold, per-session stats + channel ranking
fit on TRAINING trials only. Feature layout: EEG 310 = 62ch x 5 bands, index = ch*5+band."""
import os, pickle, warnings, numpy as np
warnings.filterwarnings("ignore")
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "results", "workstation"); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("SEEDV_DATA", os.path.join(_REPO, "data"))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
rng=np.random.RandomState(0)
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
def induct(Xe,Xy,S,tm):
    Xe=Xe.copy(); Xy=Xy.copy()
    for s in np.unique(S):
        m=S==s; src=m & tm
        if src.sum()<2: src=m
        for X in (Xe,Xy):
            mu=np.nanmean(X[src],0); sd=np.nanstd(X[src],0); X[m]=np.nan_to_num((X[m]-mu)/(sd+1e-6))
    return Xe.astype(np.float32),Xy.astype(np.float32)

# ---- 1. channel selection (inductive) ----
KS=[5,10,15,20,31,62]; chanacc={k:[] for k in KS}
for s in range(1,17):
    Xe,Xy,Y,G,S=DAT[s]
    for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=0).split(Xe,Y,G):
        tm=np.zeros(len(Y),bool); tm[tr]=True; Xe_n,Xy_n=induct(Xe,Xy,S,tm)
        F,_=f_classif(Xe_n[tr],Y[tr]); F=np.nan_to_num(F).reshape(62,5).mean(1); order=np.argsort(-F)
        for k in KS:
            ch=order[:k]; idx=np.concatenate([ch*5+b for b in range(5)])
            Xtr=np.concatenate([Xe_n[tr][:,idx],Xy_n[tr]],1); Xte=np.concatenate([Xe_n[te][:,idx],Xy_n[te]],1)
            chanacc[k].append((LogisticRegression(max_iter=500).fit(Xtr,Y[tr]).predict(Xte)==Y[te]).mean())
print("=== channel selection (inductive) ===")
full=np.mean(chanacc[62])*100
for k in KS: a=np.mean(chanacc[k])*100; print(f"  {k:2d} ch: {a:5.2f}%  ({a/full*100:.1f}% of full)")

# ---- 2. 5-seed LogReg stability (inductive) ----
seedm=[]
for seed in range(5):
    accs=[]
    for s in range(1,17):
        Xe,Xy,Y,G,S=DAT[s]
        for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(Xe,Y,G):
            tm=np.zeros(len(Y),bool); tm[tr]=True; Xe_n,Xy_n=induct(Xe,Xy,S,tm); Xc=np.concatenate([Xe_n,Xy_n],1)
            accs.append((LogisticRegression(max_iter=500).fit(Xc[tr],Y[tr]).predict(Xc[te])==Y[te]).mean())
    seedm.append(np.mean(accs)*100)
ci=1.96*np.std(seedm)/np.sqrt(5)
print(f"\n=== 5-seed LogReg (inductive): {np.mean(seedm):.2f}% +/- {ci:.2f} (95% CI), seed SD {np.std(seedm):.2f} ===")

# ---- 3. permutation importance (inductive) ----
bands=["Delta","Theta","Alpha","Beta","Gamma"]; drop={b:[] for b in bands}; drop["Eye"]=[]
for s in range(1,17):
    Xe,Xy,Y,G,S=DAT[s]
    for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=0).split(Xe,Y,G):
        tm=np.zeros(len(Y),bool); tm[tr]=True; Xe_n,Xy_n=induct(Xe,Xy,S,tm)
        Xc=np.concatenate([Xe_n,Xy_n],1); clf=LogisticRegression(max_iter=500).fit(Xc[tr],Y[tr])
        base=(clf.predict(Xc[te])==Y[te]).mean()
        for bi,b in enumerate(bands):
            Xp=Xc[te].copy(); cols=[c*5+bi for c in range(62)]
            Xp[:,cols]=Xp[rng.permutation(len(Xp))][:,cols]; drop[b].append(base-(clf.predict(Xp)==Y[te]).mean())
        Xp=Xc[te].copy(); Xp[:,310:]=Xp[rng.permutation(len(Xp))][:,310:]; drop["Eye"].append(base-(clf.predict(Xp)==Y[te]).mean())
print("\n=== permutation importance (inductive, accuracy drop %) ===")
for b in bands+["Eye"]: print(f"  {b:6s}: {np.mean(drop[b])*100:5.2f}")
# save
with open(os.path.join(_OUT, "RESULTS_inductive_fast.md"),"w") as f:
    f.write("CHANNELS(inductive): "+" | ".join(f"{k}ch {np.mean(chanacc[k])*100:.2f}" for k in KS)+"\n")
    f.write(f"5-SEED LogReg(inductive): {np.mean(seedm):.2f} +/- {ci:.2f}\n")
    f.write("PERM(inductive): "+" | ".join(f"{b} {np.mean(drop[b])*100:.2f}" for b in bands+["Eye"])+"\n")
print("\nsaved RESULTS_inductive_fast.md")
