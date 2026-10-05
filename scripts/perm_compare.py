# -*- coding: utf-8 -*-
"""Matched group-wise permutation importance under BOTH normalisation regimes, so the
transductive-vs-inductive interpretation change is a fair comparison: identical grouping
(each EEG band = all 62 channels of that band; eye = the whole 33-dim group), identical
model (LogReg), trial-independent folds, 5 permutation repeats (mean +/- SD accuracy drop)."""
import os, pickle, warnings, numpy as np
warnings.filterwarnings("ignore")
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "results", "workstation"); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("SEEDV_DATA", os.path.join(_REPO, "data"))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
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
def norm(Xe,Xy,S,tm,mode):
    Xe=Xe.copy(); Xy=Xy.copy()
    for s in np.unique(S):
        m=S==s; src=m if mode=="trans" else (m & tm)
        if src.sum()<2: src=m
        for X in (Xe,Xy):
            mu=np.nanmean(X[src],0); sd=np.nanstd(X[src],0); X[m]=np.nan_to_num((X[m]-mu)/(sd+1e-6))
    return Xe.astype(np.float32),Xy.astype(np.float32)
bands=["Delta","Theta","Alpha","Beta","Gamma"]; groups=bands+["Eye(33)"]; REP=5
res={}
for mode in ["trans","induct"]:
    drop={g:[] for g in groups}
    for s in range(1,17):
        Xe,Xy,Y,G,S=DAT[s]
        for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=0).split(Xe,Y,G):
            tm=np.zeros(len(Y),bool); tm[tr]=True; Xe_n,Xy_n=norm(Xe,Xy,S,tm,mode)
            Xc=np.concatenate([Xe_n,Xy_n],1); clf=LogisticRegression(max_iter=500).fit(Xc[tr],Y[tr])
            base=(clf.predict(Xc[te])==Y[te]).mean(); Xte=Xc[te]
            for bi,b in enumerate(bands):
                cols=[c*5+bi for c in range(62)]; ds=[]
                for _ in range(REP):
                    Xp=Xte.copy(); Xp[:,cols]=Xte[rng.permutation(len(Xte))][:,cols]; ds.append(base-(clf.predict(Xp)==Y[te]).mean())
                drop[b].append(np.mean(ds))
            ds=[]
            for _ in range(REP):
                Xp=Xte.copy(); Xp[:,310:]=Xte[rng.permutation(len(Xte))][:,310:]; ds.append(base-(clf.predict(Xp)==Y[te]).mean())
            drop["Eye(33)"].append(np.mean(ds))
    res[mode]={g:(np.mean(drop[g])*100,np.std(drop[g])*100) for g in groups}
L=["=== Matched group-wise permutation importance (acc drop %, mean +/- SD; 5 repeats) ===",
   "Note: each EEG band group = 62 channels; eye group = 33 features (larger group -> caution).",
   f"{'Group':10s}{'transductive':>18s}{'inductive':>16s}"]
for g in sorted(groups,key=lambda g:-res['induct'][g][0]):
    t=res['trans'][g]; i=res['induct'][g]; L.append(f"{g:10s}{t[0]:8.2f}+/-{t[1]:4.2f}   {i[0]:8.2f}+/-{i[1]:4.2f}")
txt="\n".join(L); print(txt,flush=True); open(os.path.join(_OUT, "RESULTS_perm_compare.md"),"w").write(txt+"\n")
