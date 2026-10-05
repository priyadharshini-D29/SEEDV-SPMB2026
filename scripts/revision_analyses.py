# -*- coding: utf-8 -*-
"""Revision analyses for IEEE SPMB 2026 p055 (reviewer requests).

1. Channel-selection stability: re-runs the training-only ANOVA channel ranking of
   inductive_fast.py (same folds, seed 0) and records which electrodes enter the
   top-k montage in each of the 48 outer folds.
2. Per-class one-vs-rest precision, recall, specificity, F1 and the row-normalized
   confusion matrix from the pooled fusion confusion counts.
3. Class balance: segments and trials per class, per subject.
"""
import json, os, pickle, warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("SEEDV_DATA", os.path.join(HERE, "data"))
OUT = os.path.join(HERE, "results", "revision", "revision_analyses")
os.makedirs(OUT, exist_ok=True)

# SEED-family 62-channel order (ESI NeuroScan cap, as distributed with SEED/SEED-V).
CH = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
      "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
      "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7",
      "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
      "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
EMO = ["Disgust", "Fear", "Sad", "Neutral", "Happy"]
assert len(CH) == 62


def load(sub):
    de = np.load(os.path.join(DATA, "EEG_DE_features", f"{sub}_123.npz"), allow_pickle=True)
    ey = np.load(os.path.join(DATA, "Eye_movement_features", f"{sub}_123.npz"), allow_pickle=True)
    dd = pickle.loads(de["data"].tobytes()); dl = pickle.loads(de["label"].tobytes())
    ed = pickle.loads(ey["data"].tobytes())
    Xe, Xy, Y, G, S, L = [], [], [], [], [], []
    for t in sorted(dd.keys()):
        e = np.asarray(dd[t], np.float32); y = np.asarray(ed[t], np.float32)
        lab = np.asarray(dl[t]).astype(int).ravel()
        n = min(e.shape[0], y.shape[0], lab.shape[0])
        L.append((t, e.shape[0], y.shape[0], lab.shape[0], int(lab[0])))
        Xe.append(e[:n]); Xy.append(y[:n]); Y.append(lab[:n]); G.append(np.full(n, t)); S.append(np.full(n, t // 15))
    return (np.concatenate(Xe), np.concatenate(Xy), np.concatenate(Y), np.concatenate(G),
            np.concatenate(S), L)


def induct(Xe, Xy, S, tm):
    Xe = Xe.copy(); Xy = Xy.copy()
    for s in np.unique(S):
        m = S == s; src = m & tm
        if src.sum() < 2:
            src = m
        for X in (Xe, Xy):
            mu = np.nanmean(X[src], 0); sd = np.nanstd(X[src], 0)
            X[m] = np.nan_to_num((X[m] - mu) / (sd + 1e-6))
    return Xe.astype(np.float32), Xy.astype(np.float32)


DAT = {s: load(s) for s in range(1, 17)}

# ---- 3. class balance and trial-length matching ----
bal = {}
mism = []
for s, (Xe, Xy, Y, G, S, L) in DAT.items():
    bal[s] = {EMO[c]: int((Y == c).sum()) for c in range(5)}
    bal[s]["trials_per_class"] = {EMO[c]: sum(1 for l in L if l[4] == c) for c in range(5)}
    for t, ne, ny, nl, c in L:
        if not (ne == ny == nl):
            mism.append((s, t, ne, ny, nl))
seg_per_trial = [l[1] for s in DAT for l in DAT[s][5]]
summary = {
    "segments_total": int(sum(len(DAT[s][2]) for s in DAT)),
    "segments_per_class": {EMO[c]: int(sum((DAT[s][2] == c).sum() for s in DAT)) for c in range(5)},
    "trials_per_class_per_subject": bal[1]["trials_per_class"],
    "segments_per_trial_min_median_max": [int(np.min(seg_per_trial)), float(np.median(seg_per_trial)), int(np.max(seg_per_trial))],
    "segments_per_subject_min_max": [int(min(len(DAT[s][2]) for s in DAT)), int(max(len(DAT[s][2]) for s in DAT))],
    "trials_with_length_mismatch": len(mism),
    "mismatch_examples": mism[:10],
    "max_segments_dropped_in_a_trial": int(max([max(a[2:]) - min(a[2:]) for a in mism], default=0)),
    "total_segments_dropped": int(sum(max(a[2:]) - min(a[2:]) for a in mism)),
}

# ---- 1. channel-selection stability (identical folds/ranking to inductive_fast.py) ----
KS = [5, 10, 15, 20, 31, 62]
counts = {k: np.zeros(62, int) for k in KS}
acc = {k: [] for k in KS}
sel10 = np.zeros((16, 3, 62), bool)       # participant x fold x electrode, 10-channel montage
nfold = 0
for s in range(1, 17):
    fi = -1
    Xe, Xy, Y, G, S, _ = DAT[s]
    for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(Xe, Y, G):
        nfold += 1; fi += 1
        tm = np.zeros(len(Y), bool); tm[tr] = True
        Xe_n, Xy_n = induct(Xe, Xy, S, tm)
        F, _ = f_classif(Xe_n[tr], Y[tr]); F = np.nan_to_num(F).reshape(62, 5).mean(1)
        order = np.argsort(-F)
        sel10[s - 1, fi, order[:10]] = True
        for k in KS:
            ch = order[:k]; counts[k][ch] += 1
            idx = np.concatenate([ch * 5 + b for b in range(5)])
            Xtr = np.concatenate([Xe_n[tr][:, idx], Xy_n[tr]], 1)
            Xte = np.concatenate([Xe_n[te][:, idx], Xy_n[te]], 1)
            acc[k].append((LogisticRegression(max_iter=500).fit(Xtr, Y[tr]).predict(Xte) == Y[te]).mean())

chan = {"n_folds": nfold, "accuracy_pct": {k: round(float(np.mean(acc[k]) * 100), 2) for k in KS}}
# per-participant consistency: electrode selected in >= 2 of that participant's 3 folds
stable_pp = (sel10.sum(1) >= 2)                       # 16 x 62
chan["top10_participants_with_channel_in_>=2of3_folds"] = {CH[i]: int(stable_pp[:, i].sum())
                                                          for i in np.argsort(-stable_pp.sum(0))[:15]}
# mean pairwise Jaccard overlap of fold-wise 10-channel sets (within and across participants)
F = sel10.reshape(48, 62).astype(int)
inter = F @ F.T; union = 20 - inter
J = inter / union
pid = np.repeat(np.arange(16), 3)
same = (pid[:, None] == pid[None, :]) & ~np.eye(48, dtype=bool)
diff = pid[:, None] != pid[None, :]
chan["jaccard_within_participant"] = round(float(J[same].mean()), 3)
chan["jaccard_across_participants"] = round(float(J[diff].mean()), 3)
np.save(os.path.join(OUT, "sel10_participant_fold_channel.npy"), sel10)
np.save(os.path.join(OUT, "channel_counts_by_k.npy"), np.stack([counts[k] for k in KS]))
for k in (5, 10):
    o = np.argsort(-counts[k], kind="stable")
    chan[f"top{k}_by_selection_frequency"] = [(CH[i], int(counts[k][i]), round(100 * counts[k][i] / nfold, 1))
                                             for i in o[:15] if counts[k][i] > 0]
    chan[f"top{k}_distinct_channels_ever_selected"] = int((counts[k] > 0).sum())
    chan[f"top{k}_channels_selected_in_>=50pct_folds"] = [CH[i] for i in o if counts[k][i] >= nfold / 2]
    # mean pairwise Jaccard overlap between fold-wise top-k sets is not recoverable from counts,
    # so report the frequency profile only.

# ---- 2. per-class one-vs-rest metrics from pooled fusion confusion counts ----
cm = np.loadtxt(os.path.join(HERE, "results", "cloud_instance", "spmb_results", "spmb_confusion_fusion_counts.csv"),
                delimiter=",", skiprows=1, usecols=range(1, 6))
N = cm.sum()
per = {}
for c in range(5):
    tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = N - tp - fn - fp
    p = tp / (tp + fp); r = tp / (tp + fn); sp = tn / (tn + fp); f1 = 2 * p * r / (p + r)
    per[EMO[c]] = {"precision": round(100 * p, 1), "recall": round(100 * r, 1),
                   "specificity": round(100 * sp, 1), "f1": round(100 * f1, 1), "support": int(cm[c].sum())}
rown = np.round(100 * cm / cm.sum(1, keepdims=True), 1)

res = {"class_balance": summary, "per_subject_segments": bal, "channel_selection": chan,
       "per_class_fusion": per, "confusion_rownorm_pct": rown.tolist(),
       "pooled_accuracy_check_pct": round(100 * np.trace(cm) / N, 2)}
with open(os.path.join(OUT, "revision_analyses.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_subject_segments"}, indent=1))
