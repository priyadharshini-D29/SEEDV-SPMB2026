# -*- coding: utf-8 -*-
"""FINAL audited cross-subject LOSO by normalisation access, addressing the reviewer:
 - all conditions on the COMMON last-50% eval window (valid paired tests);
 - NESTED selection of the online prior pseudo-count n0 (inner 4-fold over training
   subjects only -> no test-set tuning); pre-registered n0=20 reported alongside;
 - paired subject-level stats: mean diff, paired t, Wilcoxon, Holm across comparisons,
   bootstrap 95% CI, #subjects improved;
 - learning curve: online accuracy by session decile (full session, causal);
 - per-subject inductive/online/transductive for a subject-level figure.
Ordering verified separately: trial keys 0..44, 15 trials/session, rows aligned, single-emotion trials."""
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
    Xe=np.concatenate([RAW[s][0] for s in subs]); Xy=np.concatenate([RAW[s][1] for s in subs]); Y=np.concatenate([RAW[s][2] for s in subs])
    return np.concatenate([Xe,Xy],1),Y
def prior(subs):
    Xe=np.concatenate([RAW[s][0] for s in subs]); Xy=np.concatenate([RAW[s][1] for s in subs])
    return np.nanmean(Xe,0),np.nanstd(Xe,0),np.nanmean(Xy,0),np.nanstd(Xy,0)
def online_norm(X,mu0,sd0,n0):
    n,d=X.shape; out=np.empty_like(X); s1=np.zeros(d); s2=np.zeros(d); cnt=0; v0=sd0**2
    for i in range(n):
        mu=(n0*mu0+s1)/(n0+cnt); var=(n0*(v0+mu0**2)+s2)/(n0+cnt)-mu**2
        sd=np.sqrt(np.clip(var,1e-8,None)); out[i]=(X[i]-mu)/(sd+1e-6)
        s1+=X[i]; s2+=X[i]**2; cnt+=1
    return np.nan_to_num(out).astype(np.float32)

def online_pred_session(clf,xe_s,xy_s,me,se,my,sy,n0):
    Xo=np.concatenate([online_norm(xe_s,me,se,n0),online_norm(xy_s,my,sy,n0)],1)
    return clf.predict(Xo)

def online_acc_lasthalf(clf,sub,me,se,my,sy,n0):
    xe,xy,y,S=RAW[sub]; h=0;t=0
    for s in np.unique(S):
        m=np.where(S==s)[0]; half=len(m)//2; pred=online_pred_session(clf,xe[m],xy[m],me,se,my,sy,n0)
        h+=(pred[half:]==y[m][half:]).sum(); t+=len(m)-half
    return h/t

N0S=[20,50,100]; DECILES=np.arange(0,1.01,0.1)
subs_all=list(range(1,17))
R={k:[] for k in ["inductive","cal20_chrono","cal20_repr","online_prereg","online_nested","transductive"]}
pred_online={}  # for learning curve & subject figure
nested_pick=[]
lc=np.zeros((16,10)); lc_n=np.zeros((16,10))
t0=time.time()
for oi,tst in enumerate(subs_all):
    trs=[s for s in subs_all if s!=tst]
    XA,YA=tr_aligned(trs); clfA=LogisticRegression(max_iter=500).fit(XA,YA)
    XB,YB=tr_raw(trs); sc=StandardScaler().fit(XB); clfB=LogisticRegression(max_iter=500).fit(sc.transform(XB),YB)
    me,se,my,sy=prior(trs)
    # ---- nested n0 selection: inner 4-fold over the 15 training subjects ----
    inner=[trs[i::4] for i in range(4)]                       # 4 disjoint subject folds
    n0score={n0:[] for n0 in N0S}
    for vf in inner:
        itr=[s for s in trs if s not in vf]
        XAi,YAi=tr_aligned(itr); clfAi=LogisticRegression(max_iter=500).fit(XAi,YAi)
        mei,sei,myi,syi=prior(itr)
        for n0 in N0S:
            n0score[n0].append(np.mean([online_acc_lasthalf(clfAi,vs,mei,sei,myi,syi,n0) for vs in vf]))
    best_n0=max(N0S,key=lambda n0:np.mean(n0score[n0])); nested_pick.append(best_n0)
    # ---- evaluate all conditions on COMMON last-50% window ----
    xe,xy,y,S=RAW[tst]
    hit={k:0 for k in R}; tot=0
    # cache online predictions (n0=20 pre-reg, and nested) over full session for LC/subject fig
    po20=[]; poN=[]; ytrue=[]
    for s in np.unique(S):
        m=np.where(S==s)[0]; n=len(m); half=n//2; ev=m[half:]; cal=m[:half]; Ye=y[ev]; tot+=len(ev)
        # inductive
        hit["inductive"]+=(clfB.predict(sc.transform(np.concatenate([xe[ev],xy[ev]],1)))==Ye).sum()
        # transductive (full-session stats)
        Xt=np.concatenate([zw(xe[ev],np.nanmean(xe[m],0),np.nanstd(xe[m],0)),zw(xy[ev],np.nanmean(xy[m],0),np.nanstd(xy[m],0))],1)
        hit["transductive"]+=(clfA.predict(Xt)==Ye).sum()
        # chronological calibration 20% (first 20% of session)
        k=max(1,int(0.2*n)); c=m[:k]
        Xc=np.concatenate([zw(xe[ev],np.nanmean(xe[c],0),np.nanstd(xe[c],0)),zw(xy[ev],np.nanmean(xy[c],0),np.nanstd(xy[c],0))],1)
        hit["cal20_chrono"]+=(clfA.predict(Xc)==Ye).sum()
        # representative (random) 20% from the first-50% region (secondary, not headline)
        kk=max(1,int(0.2*n)); ridx=cal[rng.permutation(len(cal))[:kk]] if len(cal)>=kk else cal
        Xr=np.concatenate([zw(xe[ev],np.nanmean(xe[ridx],0),np.nanstd(xe[ridx],0)),zw(xy[ev],np.nanmean(xy[ridx],0),np.nanstd(xy[ridx],0))],1)
        hit["cal20_repr"]+=(clfA.predict(Xr)==Ye).sum()
        # online pre-reg n0=20 and nested, on last-50% window
        p20=online_pred_session(clfA,xe[m],xy[m],me,se,my,sy,20)
        pN=online_pred_session(clfA,xe[m],xy[m],me,se,my,sy,best_n0)
        hit["online_prereg"]+=(p20[half:]==Ye).sum(); hit["online_nested"]+=(pN[half:]==Ye).sum()
        po20.append(p20); poN.append(pN); ytrue.append(y[m])
        # learning curve (full session, n0=20): accuracy by decile of session progress
        prog=(np.arange(n))/n
        for di in range(10):
            dm=(prog>=DECILES[di])&(prog<DECILES[di+1])
            if dm.sum()>0: lc[oi,di]+=(p20[dm]==y[m][dm]).sum(); lc_n[oi,di]+=dm.sum()
    for k in R: R[k].append(hit[k]/tot)
    pred_online[tst]=(np.concatenate(po20),np.concatenate(poN),np.concatenate(ytrue))
    print(f"  S{tst:02d}: ind {R['inductive'][-1]*100:4.1f} | on20 {R['online_prereg'][-1]*100:4.1f} | onNest(n0={best_n0}) {R['online_nested'][-1]*100:4.1f} | trans {R['transductive'][-1]*100:4.1f}  ({time.time()-t0:.0f}s)",flush=True)

for k in R: R[k]=np.array(R[k])
# full-session online (n0=20)
onlinefull=np.array([ (pred_online[s][0]==pred_online[s][2]).mean() for s in subs_all])

def boot_ci(d,B=10000):
    idx=rng.randint(0,len(d),(B,len(d))); m=d[idx].mean(1); return np.percentile(m,2.5),np.percentile(m,97.5)
def compare(a,b,name):
    d=(a-b)*100; _,pt=st.ttest_rel(a,b)
    try: _,pw=st.wilcoxon(a,b)
    except: pw=np.nan
    lo,hi=boot_ci(d); return dict(name=name,diff=d.mean(),lo=lo,hi=hi,pt=pt,pw=pw,win=int((a>b).sum()))
comps=[compare(R["online_prereg"],R["inductive"],"online vs inductive"),
       compare(R["online_prereg"],R["cal20_chrono"],"online vs chrono-cal20"),
       compare(R["online_prereg"],R["cal20_repr"],"online vs repr-cal20"),
       compare(R["online_prereg"],R["transductive"],"online vs transductive")]
# Holm across the 4 (use Wilcoxon p)
ps=sorted([(c["pw"],c["name"]) for c in comps]); mH={}
_run=0.0
for rank,(p,nm) in enumerate(ps): _run=max(_run,min(1.0,p*(len(ps)-rank))); mH[nm]=_run   # step-down with running maximum

L=["=== FINAL LOSO by normalisation access (16 subjects, COMMON last-50% window) ==="]
def ln(n,a): return f"{n:26s}: {a.mean()*100:5.2f} +/- {a.std()*100:4.1f}"
for nm,k in [("Inductive","inductive"),("Calibration chrono-20%","cal20_chrono"),("Calibration repr-20% (2ndary)","cal20_repr"),
             ("Online n0=20 (pre-reg)","online_prereg"),("Online n0=nested","online_nested"),("Transductive","transductive")]:
    L.append(ln(nm,R[k]))
L.append(f"{'Online full-session (n0=20)':26s}: {onlinefull.mean()*100:5.2f} +/- {onlinefull.std()*100:4.1f}")
L.append(f"\nNested n0 picks across 16 folds: {nested_pick}  (mode={max(set(nested_pick),key=nested_pick.count)})")
L.append("\nPaired comparisons (online n0=20, last-50% window; Holm on Wilcoxon):")
for c in comps:
    L.append(f"  {c['name']:22s}: diff {c['diff']:+5.2f} pts | 95%CI [{c['lo']:+.2f},{c['hi']:+.2f}] | t p={c['pt']:.2e} | Wilcoxon p={c['pw']:.2e} | Holm p={mH[c['name']]:.2e} | improved {c['win']}/16")
L.append("\nLearning curve (online n0=20, accuracy % by session decile):")
dec=100*lc.sum(0)/np.maximum(lc_n.sum(0),1)
L.append("  "+" | ".join(f"{int(DECILES[i]*100)}-{int(DECILES[i+1]*100)}%:{dec[i]:4.1f}" for i in range(10)))
L.append("\nPer-subject (inductive / online n0=20 last-50% / transductive):")
for i,s in enumerate(subs_all):
    L.append(f"  S{s:02d}: {R['inductive'][i]*100:5.1f} / {R['online_prereg'][i]*100:5.1f} / {R['transductive'][i]*100:5.1f}")
txt="\n".join(L); print("\n"+txt,flush=True)
open(os.path.join(_OUT, "RESULTS_loso_final.md"),"w",encoding="utf-8").write(txt+"\n")
np.savez(os.path.join(_OUT, "loso_final_arrays.npz"),inductive=R["inductive"],online=R["online_prereg"],
         transductive=R["transductive"],onlinefull=onlinefull,decile=dec,subs=np.array(subs_all))
print(f"\nsaved RESULTS_loso_final.md + loso_final_arrays.npz ({time.time()-t0:.0f}s)",flush=True)
