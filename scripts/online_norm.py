# -*- coding: utf-8 -*-
"""Honest DEPLOYABLE cross-subject: causal / online normalisation.
Each test sample is normalised using ONLY past samples of its session (a causal running
mean/std), warm-started with a training-derived prior (pseudo-count n0). No future samples,
no labels -> streaming-deployable. Compared on the common last-50% window AND full session.
Classifier = clfA (trained in per-session-z-norm space, as in the aligned pipeline)."""
import os, pickle, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "results", "workstation"); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("SEEDV_DATA", os.path.join(_REPO, "data"))
from sklearn.linear_model import LogisticRegression
import scipy.stats as st

def load(sub):
    de=np.load(os.path.join(DATA,"EEG_DE_features",f"{sub}_123.npz"),allow_pickle=True)
    ey=np.load(os.path.join(DATA,"Eye_movement_features",f"{sub}_123.npz"),allow_pickle=True)
    dd=pickle.loads(de["data"].tobytes()); dl=pickle.loads(de["label"].tobytes()); ed=pickle.loads(ey["data"].tobytes())
    Xe,Xy,Y,S=[],[],[],[]
    for t in sorted(dd.keys()):
        e=np.asarray(dd[t],np.float32); y=np.asarray(ed[t],np.float32); lab=np.asarray(dl[t]).astype(int).ravel()
        n=min(e.shape[0],y.shape[0],lab.shape[0])
        Xe.append(e[:n]);Xy.append(y[:n]);Y.append(lab[:n]);S.append(np.full(n,t//15))
    return np.concatenate(Xe),np.concatenate(Xy),np.concatenate(Y),np.concatenate(S)
def zw(X,mu,sd): return np.nan_to_num((X-mu)/(sd+1e-6)).astype(np.float32)
def sessnorm(X,S):
    X=X.copy()
    for s in np.unique(S):
        m=S==s; X[m]=zw(X[m],np.nanmean(X[m],0),np.nanstd(X[m],0))
    return X
RAW={s:load(s) for s in range(1,17)}
def tr_aligned(subs):
    Xe,Xy,Y=[],[],[]
    for s in subs:
        xe,xy,y,S=RAW[s]; Xe.append(sessnorm(xe,S)); Xy.append(sessnorm(xy,S)); Y.append(y)
    return np.concatenate([np.concatenate(Xe),np.concatenate(Xy)],1),np.concatenate(Y)
def prior(subs):
    Xe=np.concatenate([RAW[s][0] for s in subs]); Xy=np.concatenate([RAW[s][1] for s in subs])
    return (np.nanmean(Xe,0),np.nanstd(Xe,0),np.nanmean(Xy,0),np.nanstd(Xy,0))

def online_norm_session(X,mu0,sd0,n0):
    """Return per-sample causal-normalised X. Sample i uses prior + running stats over 0..i-1;
    for i=0 only the prior is available (n0 pseudo-count anchors early samples)."""
    n,d=X.shape; out=np.empty_like(X)
    s1=np.zeros(d); s2=np.zeros(d); cnt=0
    v0=sd0**2
    for i in range(n):
        # blended stats from prior (n0 pseudo-obs) + running (cnt past obs)
        mu=(n0*mu0+s1)/(n0+cnt); var=(n0*(v0+mu0**2)+s2)/(n0+cnt)-mu**2
        sd=np.sqrt(np.clip(var,1e-8,None))
        out[i]=(X[i]-mu)/(sd+1e-6)
        s1+=X[i]; s2+=X[i]**2; cnt+=1     # update AFTER predicting (causal)
    return np.nan_to_num(out).astype(np.float32)

N0=[20,50,100]; t0=time.time()
res={f"online_n{n0}_half":[] for n0 in N0}; res.update({f"online_n{n0}_full":[] for n0 in N0})
for tst in range(1,17):
    trs=[s for s in range(1,17) if s!=tst]
    XA,YA=tr_aligned(trs); clfA=LogisticRegression(max_iter=500).fit(XA,YA)
    me,se,my,sy=prior(trs)
    xe,xy,y,S=RAW[tst]
    for n0 in N0:
        h_half=t_half=h_full=t_full=0
        for s in np.unique(S):
            m=np.where(S==s)[0]; n=len(m); half=n//2
            Xe_o=online_norm_session(xe[m],me,se,n0); Xy_o=online_norm_session(xy[m],my,sy,n0)
            Xo=np.concatenate([Xe_o,Xy_o],1); pred=clfA.predict(Xo); Ym=y[m]
            h_full+=(pred==Ym).sum(); t_full+=n
            h_half+=(pred[half:]==Ym[half:]).sum(); t_half+=n-half
        res[f"online_n{n0}_full"].append(h_full/t_full); res[f"online_n{n0}_half"].append(h_half/t_half)
    print(f"  S{tst:02d}: online(n50) full {res['online_n50_full'][-1]*100:4.1f} | last50 {res['online_n50_half'][-1]*100:4.1f}  ({time.time()-t0:.0f}s)",flush=True)

for k in res: res[k]=np.array(res[k])
L=["=== CAUSAL/ONLINE NORMALISATION (deployable, 16 subjects) ==="]
L.append("(reference: inductive 43.2 | representative-calib20 64.8 | transductive 74.3, on last-50% window)")
for n0 in N0:
    L.append(f"online prior-n0={n0:3d}:  full-session {res[f'online_n{n0}_full'].mean()*100:5.2f} +/- {res[f'online_n{n0}_full'].std()*100:4.1f}  |  last-50% window {res[f'online_n{n0}_half'].mean()*100:5.2f} +/- {res[f'online_n{n0}_half'].std()*100:4.1f}")
txt="\n".join(L); print("\n"+txt,flush=True)
open(os.path.join(_OUT, "RESULTS_loso_online.md"),"w",encoding="utf-8").write(txt+"\n")
print(f"\nsaved RESULTS_loso_online.md ({time.time()-t0:.0f}s)",flush=True)
