# -*- coding: utf-8 -*-
"""Reviewer blocker #1: the trial-independent 'honest' result currently normalises each
session with WHOLE-session statistics, so held-out test trials contribute to the mean/std
(session-level transductive access). This re-runs the trial-grouped protocol with
TRAINING-TRIAL-ONLY per-session normalisation (truly inductive) and compares. Also emits the
per-class recall + confusion from the inductive LogReg so the figure/table match one run."""
import os, pickle, warnings, numpy as np
warnings.filterwarnings("ignore")
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "results", "workstation"); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("SEEDV_DATA", os.path.join(_REPO, "data"))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
np.random.seed(0)
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
DAT={s:load(s) for s in range(1,17)}
def norm(Xe,Xy,S,trmask,mode):
    Xe=Xe.copy(); Xy=Xy.copy()
    for s in np.unique(S):
        m=S==s
        src = m if mode=="trans" else (m & trmask)
        if src.sum()<2: src=m
        for X in (Xe,Xy):
            mu=np.nanmean(X[src],0); sd=np.nanstd(X[src],0)
            X[m]=np.nan_to_num((X[m]-mu)/(sd+1e-6))
    return np.concatenate([Xe,Xy],1).astype(np.float32)
MODELS={"LogReg":lambda:LogisticRegression(max_iter=500),"SVM-RBF":lambda:SVC(C=1,gamma="scale"),
 "LDA":LinearDiscriminantAnalysis,"KNN":lambda:KNeighborsClassifier(15),
 "RandomForest":lambda:RandomForestClassifier(200,n_jobs=-1,random_state=0),
 "GradBoost":lambda:GradientBoostingClassifier(random_state=0),"NaiveBayes":GaussianNB,
 "MLP":lambda:MLPClassifier((256,128),max_iter=300,random_state=0)}
def run(mode,collect_cm=False):
    acc={k:[] for k in MODELS}; yt_all=[]; yp_all=[]
    for s in range(1,17):
        Xe,Xy,Y,G,S=DAT[s]
        for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=0).split(Xe,Y,G):
            trmask=np.zeros(len(Y),bool); trmask[tr]=True
            Xc=norm(Xe,Xy,S,trmask,mode)
            for name,mk in MODELS.items():
                p=mk().fit(Xc[tr],Y[tr]).predict(Xc[te]); acc[name].append((p==Y[te]).mean())
                if collect_cm and name=="LogReg": yt_all+=list(Y[te]); yp_all+=list(p)
    return acc,(np.array(yt_all),np.array(yp_all))
print("running transductive (current) ...",flush=True); at,_=run("trans")
print("running inductive (training-only norm) ...",flush=True); ai,(yt,yp)=run("induct",collect_cm=True)
L=["=== TRIAL-INDEPENDENT: session-transductive (current) vs training-only inductive norm ==="]
L.append(f"{'Model':14s}{'trans%':>9s}{'induct%':>9s}{'delta':>8s}")
for k in MODELS:
    t=np.mean(at[k])*100; i=np.mean(ai[k])*100; L.append(f"{k:14s}{t:8.2f} {i:8.2f} {i-t:+7.2f}")
EMO=["Disgust","Fear","Sad","Neutral","Happy"]
cm=confusion_matrix(yt,yp); cmn=cm/cm.sum(1,keepdims=True)
L.append("\nInductive LogReg: acc %.2f | balanced-acc %.2f | macroF1 %.3f"%((yt==yp).mean()*100,balanced_accuracy_score(yt,yp)*100,f1_score(yt,yp,average="macro")))
L.append("Per-class recall (inductive, pooled): "+" | ".join(f"{EMO[i]} {100*cmn[i,i]:.1f}" for i in range(5)))
txt="\n".join(L); print("\n"+txt,flush=True)
open(os.path.join(_OUT, "RESULTS_trialindep_inductive.md"),"w",encoding="utf-8").write(txt+"\n")
np.save(os.path.join(_OUT, "cm_inductive.npy"),cmn)
print("\nsaved RESULTS_trialindep_inductive.md + cm_inductive.npy",flush=True)
