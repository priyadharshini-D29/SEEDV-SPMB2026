# -*- coding: utf-8 -*-
"""Convergence check for the two LOSO logistic regressions (max_iter=500 in loso_final.py).
For every held-out participant, the strictly inductive classifier (pooled StandardScaler) and the
session-aligned classifier (used by the causal-online and transductive conditions) are refitted with
max_iter=500 and with max_iter=20000, and latter-half accuracy is compared."""
import os, pickle, warnings, numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "results", "reproduction_check"); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("SEEDV_DATA", os.path.join(_REPO, "data"))


def load(sub):
    de = np.load(os.path.join(DATA, "EEG_DE_features", f"{sub}_123.npz"), allow_pickle=True)
    ey = np.load(os.path.join(DATA, "Eye_movement_features", f"{sub}_123.npz"), allow_pickle=True)
    dd = pickle.loads(de["data"].tobytes()); dl = pickle.loads(de["label"].tobytes()); ed = pickle.loads(ey["data"].tobytes())
    Xe, Xy, Y, S = [], [], [], []
    for t in sorted(dd.keys()):
        e = np.asarray(dd[t], np.float32); y = np.asarray(ed[t], np.float32); lab = np.asarray(dl[t]).astype(int).ravel()
        n = min(e.shape[0], y.shape[0], lab.shape[0])
        Xe.append(e[:n]); Xy.append(y[:n]); Y.append(lab[:n]); S.append(np.full(n, t // 15))
    return np.concatenate([np.concatenate(Xe), np.concatenate(Xy)], 1), np.concatenate(Y), np.concatenate(S)


def zs(X, ref):
    return np.nan_to_num((X - np.nanmean(ref, 0)) / (np.nanstd(ref, 0) + 1e-6)).astype(np.float32)


def sessnorm(X, S):
    X = X.copy()
    for s in np.unique(S):
        m = S == s; X[m] = zs(X[m], X[m])
    return X


def fit(X, Y, it):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf = LogisticRegression(max_iter=it).fit(X, Y)
    return clf, int(clf.n_iter_[0]), any(issubclass(x.category, ConvergenceWarning) for x in w)


RAW = {s: load(s) for s in range(1, 17)}
rows = []
for tst in range(1, 17):
    trs = [s for s in range(1, 17) if s != tst]
    XB = np.concatenate([RAW[s][0] for s in trs]); YB = np.concatenate([RAW[s][1] for s in trs])
    XA = np.concatenate([sessnorm(RAW[s][0], RAW[s][2]) for s in trs])
    sc = StandardScaler().fit(XB); XBs = sc.transform(XB)
    X, y, S = RAW[tst]
    ev = np.concatenate([np.where(S == s)[0][len(np.where(S == s)[0]) // 2:] for s in np.unique(S)])
    Xt = np.concatenate([zs(X[np.where(S == s)[0][len(np.where(S == s)[0]) // 2:]], X[S == s]) for s in np.unique(S)])
    r = [tst]
    for it in (500, 20000):
        cB, nB, wB = fit(XBs, YB, it); cA, nA, wA = fit(XA, YB, it)
        r += [nB, wB, 100 * (cB.predict(sc.transform(X[ev])) == y[ev]).mean(), nA, wA, 100 * (cA.predict(Xt) == y[ev]).mean()]
    rows.append(r)
    print("S%02d inductive: %d it (limit hit %s) %.2f -> %d it (limit hit %s) %.2f | aligned/transductive: %d it (%s) %.2f -> %d it (%s) %.2f"
          % (tst, r[1], r[2], r[3], r[7], r[8], r[9], r[4], r[5], r[6], r[10], r[11], r[12]), flush=True)
a = np.array(rows, float)
L = ["# LOSO logistic-regression convergence check (latter-half accuracy, 16 held-out participants)", "",
     "| Classifier | max_iter | folds stopped at the limit | mean iterations | accuracy (%) |", "|---|---:|---:|---:|---:|",
     "| Strictly inductive | 500 | %d/16 | %.0f | %.2f |" % (a[:, 2].sum(), a[:, 1].mean(), a[:, 3].mean()),
     "| Strictly inductive | 20000 | %d/16 | %.0f | %.2f |" % (a[:, 8].sum(), a[:, 7].mean(), a[:, 9].mean()),
     "| Session-aligned (transductive reference) | 500 | %d/16 | %.0f | %.2f |" % (a[:, 5].sum(), a[:, 4].mean(), a[:, 6].mean()),
     "| Session-aligned (transductive reference) | 20000 | %d/16 | %.0f | %.2f |" % (a[:, 11].sum(), a[:, 10].mean(), a[:, 12].mean()),
     "", "Largest per-participant change, inductive: %.2f points; session-aligned: %.2f points."
     % (np.abs(a[:, 9] - a[:, 3]).max(), np.abs(a[:, 12] - a[:, 6]).max())]
open(os.path.join(_OUT, "RESULTS_loso_convergence.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
