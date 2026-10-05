# -*- coding: utf-8 -*-
"""Full 12-model benchmark under TRIAL-INDEPENDENT, strictly INDUCTIVE normalisation:
per fold, per-session mean/std are fit on OUTER-TRAINING trials only and applied unchanged
to held-out trials. Compare to the session-transductive numbers (bench.py). Emits acc,
balanced-acc, macro-F1, plus LogReg confusion/per-class."""
import os, pickle, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "results", "workstation"); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("SEEDV_DATA", os.path.join(_REPO, "data"))
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
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
def induct_norm(Xe,Xy,S,trmask):
    Xe=Xe.copy(); Xy=Xy.copy()
    for s in np.unique(S):
        m=S==s; src=m & trmask
        if src.sum()<2: src=m
        for X in (Xe,Xy):
            mu=np.nanmean(X[src],0); sd=np.nanstd(X[src],0); X[m]=np.nan_to_num((X[m]-mu)/(sd+1e-6))
    return Xe.astype(np.float32),Xy.astype(np.float32)
def run_torch(mk,Xe_tr,Xy_tr,Y_tr,Xe_te,Xy_te,ep=40,lr=1e-3):
    m=mk(); opt=torch.optim.Adam(m.parameters(),lr,weight_decay=1e-4)
    xe,xy,yt=torch.tensor(Xe_tr),torch.tensor(Xy_tr),torch.tensor(Y_tr); m.train()
    for _ in range(ep): opt.zero_grad(); nn.functional.cross_entropy(m(xe,xy),yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(torch.tensor(Xe_te),torch.tensor(Xy_te)).argmax(1).numpy()
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(343,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,5))
    def forward(s,xe,xy): return s.f(torch.cat([xe,xy],1))
class CNN(nn.Module):
    def __init__(s): super().__init__(); s.c=nn.Sequential(nn.Conv1d(5,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); s.h=nn.Sequential(nn.Linear(32+33,64),nn.ReLU(),nn.Linear(64,5))
    def forward(s,xe,xy): z=s.c(xe.view(-1,62,5).transpose(1,2)).squeeze(-1); return s.h(torch.cat([z,xy],1))
class BiLSTM(nn.Module):
    def __init__(s): super().__init__(); s.l=nn.LSTM(5,32,batch_first=True,bidirectional=True); s.h=nn.Sequential(nn.Linear(64+33,64),nn.ReLU(),nn.Linear(64,5))
    def forward(s,xe,xy): o,_=s.l(xe.view(-1,62,5)); return s.h(torch.cat([o.mean(1),xy],1))
class TrEnc(nn.Module):
    def __init__(s): super().__init__(); s.proj=nn.Linear(5,32); s.enc=nn.TransformerEncoderLayer(32,4,64,batch_first=True); s.h=nn.Sequential(nn.Linear(32+33,64),nn.ReLU(),nn.Linear(64,5))
    def forward(s,xe,xy): z=s.enc(s.proj(xe.view(-1,62,5))).mean(1); return s.h(torch.cat([z,xy],1))
class CrossModal(nn.Module):
    def __init__(s):
        super().__init__(); s.proj=nn.Linear(5,32); s.enc=nn.TransformerEncoderLayer(32,4,64,batch_first=True)
        s.eye=nn.Sequential(nn.Linear(33,32),nn.ReLU()); s.cross=nn.MultiheadAttention(32,4,batch_first=True)
        s.h=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,5))
    def forward(s,xe,xy):
        e=s.enc(s.proj(xe.view(-1,62,5))); yv=s.eye(xy).unsqueeze(1); a,_=s.cross(yv,e,e)
        return s.h(torch.cat([e.mean(1),a.squeeze(1)],1))
CLASSICAL={"SVM-RBF":lambda:SVC(C=1,gamma="scale"),"LDA":LinearDiscriminantAnalysis,"LogReg":lambda:LogisticRegression(max_iter=500),
 "KNN":lambda:KNeighborsClassifier(15),"RandomForest":lambda:RandomForestClassifier(200,n_jobs=-1,random_state=0),
 "GradBoost":lambda:GradientBoostingClassifier(random_state=0),"NaiveBayes":GaussianNB}
DL={"MLP":MLP,"1D-CNN":CNN,"BiLSTM":BiLSTM,"Transformer":TrEnc,"CrossModal(ours)":CrossModal}
RES={k:{"acc":[],"f1":[],"ba":[]} for k in list(CLASSICAL)+list(DL)}; yt_all=[]; yp_all=[]; t0=time.time()
for s in range(1,17):
    Xe,Xy,Y,G,S=DAT[s]
    for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=0).split(Xe,Y,G):
        tm=np.zeros(len(Y),bool); tm[tr]=True
        Xe_n,Xy_n=induct_norm(Xe,Xy,S,tm); Xc=np.concatenate([Xe_n,Xy_n],1)
        for name,mk in CLASSICAL.items():
            p=mk().fit(Xc[tr],Y[tr]).predict(Xc[te]); RES[name]["acc"].append((p==Y[te]).mean())
            RES[name]["f1"].append(f1_score(Y[te],p,average="macro")); RES[name]["ba"].append(balanced_accuracy_score(Y[te],p))
            if name=="LogReg": yt_all+=list(Y[te]); yp_all+=list(p)
        for name,mk in DL.items():
            p=run_torch(mk,Xe_n[tr],Xy_n[tr],Y[tr],Xe_n[te],Xy_n[te]); RES[name]["acc"].append((p==Y[te]).mean())
            RES[name]["f1"].append(f1_score(Y[te],p,average="macro")); RES[name]["ba"].append(balanced_accuracy_score(Y[te],p))
    print(f"  subj {s:2d}/16 done ({time.time()-t0:.0f}s) LogReg={np.mean(RES['LogReg']['acc'])*100:.2f}",flush=True)
lines=["# Inductive benchmark (trial-independent, training-only normalisation, 16 subjects)","",f"{'Model':18s}{'Acc%':>8s}{'BalAcc%':>9s}{'MacroF1':>9s}"]
for k in sorted(RES,key=lambda k:-np.mean(RES[k]["acc"])):
    a=np.array(RES[k]["acc"]); f=np.array(RES[k]["f1"]); b=np.array(RES[k]["ba"])
    lines.append(f"{k:18s}{a.mean()*100:7.2f} {b.mean()*100:8.2f} {f.mean():8.3f}")
yt,yp=np.array(yt_all),np.array(yp_all); cmn=confusion_matrix(yt,yp)/confusion_matrix(yt,yp).sum(1,keepdims=True)
EMO=["Disgust","Fear","Sad","Neutral","Happy"]
lines.append("\nLogReg per-class recall (inductive): "+" | ".join(f"{EMO[i]} {100*cmn[i,i]:.1f}" for i in range(5)))
np.save(os.path.join(_OUT, "cm_inductive.npy"),cmn)
open(os.path.join(_OUT, "RESULTS_benchmark_inductive.md"),"w",encoding="utf-8").write("\n".join(lines)+"\n")
print("\n".join(lines)); print("\nsaved RESULTS_benchmark_inductive.md + cm_inductive.npy   %.0fs"%(time.time()-t0),flush=True)
