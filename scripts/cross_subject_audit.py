# -*- coding: utf-8 -*-
"""AUDITED LOSO normalisation experiment. Every condition is evaluated on ONE common
held-out window (the last 50% of each test session), so accuracies are directly comparable
and paired tests are valid. Calibration uses only a prefix of the FIRST 50% (disjoint from
the eval window). A random-subset calibration probes the 'prefix not representative' concern.
  - Inductive     : training statistics only (global StandardScaler + clfB).      [no test stats]
  - Calibration-k : stats from first k% of session (k<=50, within calib region, clfA).
  - Calib-rand-20 : stats from a RANDOM 20% of the first-50% region (clfA).
  - Transductive  : stats from the ENTIRE session incl. eval samples (clfA).       [full test stats]
"""
import os, pickle, time, warnings, numpy as np
warnings.filterwarnings("ignore")
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "results", "workstation"); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("SEEDV_DATA", os.path.join(_REPO, "data"))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
rng=np.random.RandomState(0)

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
def tr_raw(subs):
    Xe,Xy,Y=[],[],[]
    for s in subs:
        xe,xy,y,S=RAW[s]; Xe.append(xe); Xy.append(xy); Y.append(y)
    return np.concatenate([np.concatenate(Xe),np.concatenate(Xy)],1),np.concatenate(Y)

CAL=[10,20,30,40,50]
conds=["inductive"]+[f"cal{c}" for c in CAL]+["cal_rand20","transductive"]
R={k:[] for k in conds}; t0=time.time()
for tst in range(1,17):
    trs=[s for s in range(1,17) if s!=tst]
    XA,YA=tr_aligned(trs); clfA=LogisticRegression(max_iter=500).fit(XA,YA)
    XB,YB=tr_raw(trs); sc=StandardScaler().fit(XB); clfB=LogisticRegression(max_iter=500).fit(sc.transform(XB),YB)
    xe,xy,y,S=RAW[tst]
    hit={k:0 for k in conds}; tot=0
    for s in np.unique(S):
        m=np.where(S==s)[0]; n=len(m); half=n//2
        ev=m[half:]                       # COMMON eval window = last 50% (never used for calibration)
        calregion=m[:half]                # first 50%: calibration pool
        Ye=y[ev]; tot+=len(ev)
        # inductive (clfB, global scaler)
        hit["inductive"]+=(clfB.predict(sc.transform(np.concatenate([xe[ev],xy[ev]],1)))==Ye).sum()
        # transductive (clfA, full-session stats incl. eval)
        Xt=np.concatenate([zw(xe[ev],np.nanmean(xe[m],0),np.nanstd(xe[m],0)),zw(xy[ev],np.nanmean(xy[m],0),np.nanstd(xy[m],0))],1)
        hit["transductive"]+=(clfA.predict(Xt)==Ye).sum()
        # calibration prefixes (clfA)
        for c in CAL:
            k=max(1,int(c/100*n)); cal=m[:k]
            Xc=np.concatenate([zw(xe[ev],np.nanmean(xe[cal],0),np.nanstd(xe[cal],0)),zw(xy[ev],np.nanmean(xy[cal],0),np.nanstd(xy[cal],0))],1)
            hit[f"cal{c}"]+=(clfA.predict(Xc)==Ye).sum()
        # random-20% calibration from the first-50% region (representativeness probe)
        kk=max(1,int(0.2*n)); ridx=calregion[rng.permutation(len(calregion))[:kk]] if len(calregion)>=kk else calregion
        Xr=np.concatenate([zw(xe[ev],np.nanmean(xe[ridx],0),np.nanstd(xe[ridx],0)),zw(xy[ev],np.nanmean(xy[ridx],0),np.nanstd(xy[ridx],0))],1)
        hit["cal_rand20"]+=(clfA.predict(Xr)==Ye).sum()
    for k in conds: R[k].append(hit[k]/tot)
    print(f"  S{tst:02d}: ind {R['inductive'][-1]*100:4.1f} c20 {R['cal20'][-1]*100:4.1f} c50 {R['cal50'][-1]*100:4.1f} rnd20 {R['cal_rand20'][-1]*100:4.1f} trans {R['transductive'][-1]*100:4.1f} ({time.time()-t0:.0f}s)",flush=True)

for k in conds: R[k]=np.array(R[k])
def ln(n,a): return f"{n:16s}: {a.mean()*100:5.2f} +/- {a.std()*100:4.1f}"
L=["=== AUDITED LOSO by normalisation access (16 subjects, COMMON last-50% eval window) ==="]
L.append(ln("Inductive",R["inductive"]))
for c in CAL: L.append(ln(f"Calibration-{c}%",R[f"cal{c}"]))
L.append(ln("Calib-random20%",R["cal_rand20"]))
L.append(ln("Transductive",R["transductive"]))
def pt(a,b):
    _,p=st.ttest_rel(a,b); return p
L.append("")
L.append(f"Transductive vs Inductive : {(R['transductive'].mean()-R['inductive'].mean())*100:+.2f} pts (paired t p={pt(R['transductive'],R['inductive']):.2e})")
L.append(f"Transductive vs Calib-50% : {(R['transductive'].mean()-R['cal50'].mean())*100:+.2f} pts (paired t p={pt(R['transductive'],R['cal50']):.2e})")
L.append(f"Calib-50% vs Inductive    : {(R['cal50'].mean()-R['inductive'].mean())*100:+.2f} pts (paired t p={pt(R['cal50'],R['inductive']):.2e})")
L.append(f"Calib-random20% vs Calib-20% (prefix representativeness): {(R['cal_rand20'].mean()-R['cal20'].mean())*100:+.2f} pts (paired t p={pt(R['cal_rand20'],R['cal20']):.2e})")
txt="\n".join(L); print("\n"+txt,flush=True)
open(os.path.join(_OUT, "RESULTS_loso_audit.md"),"w",encoding="utf-8").write(txt+"\n")
print(f"\nsaved RESULTS_loso_audit.md ({time.time()-t0:.0f}s)",flush=True)
