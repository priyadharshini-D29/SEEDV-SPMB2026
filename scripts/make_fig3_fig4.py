# -*- coding: utf-8 -*-
"""Figures 3 and 4 of the paper.
Fig. 3 (full width): (A) accuracy vs. montage size, (B) electrode selection stability for the
10-channel montage across folds and participants, (C) group permutation importance.
Fig. 4 (column width): row-normalized confusion matrix + one-vs-rest metrics heatmaps.
Values come from results/revision/revision_analyses (see revision_analyses.py)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "revision", "revision_analyses", "revision_analyses.json"), encoding="utf8"))
sel10 = np.load(os.path.join(ROOT, "results", "revision", "revision_analyses", "sel10_participant_fold_channel.npy"))
from revision_analyses_channels import CH, EMO   # noqa: E402

# palette (dataviz reference instance): categorical slots + blue/orange sequential ramps
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_ORANGE = ["#fdf5f1", "#fbdccd", "#f6b89c", "#f1936b", "#eb6834", "#c9501f", "#9c3b13"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


# ------------------------------------------------------------------ Figure 3
ch = R["channel_selection"]
ks = [5, 10, 15, 20, 31, 62]
acc = [ch["accuracy_pct"][str(k)] for k in ks]

nfold = ch["n_folds"]
fold_pct = 100 * sel10.reshape(48, 62).sum(0) / nfold
pp_pct = 100 * (sel10.sum(1) >= 2).sum(0) / 16
order = np.argsort(-fold_pct, kind="stable")[:12]

perm = [("Eye", 11.27), ("Gamma", 9.48), ("Alpha", 8.81), ("Beta", 7.27), ("Theta", 7.00), ("Delta", 6.46)]

fig = plt.figure(figsize=(6.5, 2.05))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.75, 0.95], left=0.075, right=0.99, bottom=0.24, top=0.86, wspace=0.38)

ax = fig.add_subplot(gs[0])
ax.plot(ks, acc, color=BLUE, lw=2, marker="o", ms=4, mec="white", mew=0.8, zorder=3)
ax.axhline(acc[-1], color=INK2, lw=0.8, ls=(0, (3, 2)))
for k, a in zip(ks, acc):
    if k in (5, 10, 31):
        ax.annotate(f"{a:.1f}", (k, a), xytext=(7, -8) if k != 31 else (5, -10), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK)
ax.text(4.5, acc[-1] + 0.5, "all 62 channels: 69.7", ha="left", va="bottom", fontsize=6.5, color=INK2)
ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 31, 62]); ax.set_xticklabels(["5", "10", "20", "31", "62"]); ax.minorticks_off(); ax.set_xlim(4.2, 75); ax.set_ylim(57, 72)
ax.set_xlabel("EEG channels kept (log scale)"); ax.set_ylabel("Accuracy (%)")
ax.set_title("(A) Montage size", fontsize=7.5, weight="bold", loc="left", color=INK)
tidy(ax)

ax = fig.add_subplot(gs[1])
x = np.arange(len(order)); w = 0.4
ax.bar(x - w / 2 - 0.01, fold_pct[order], w, color=BLUE, label="% of 48 folds", zorder=3)
ax.bar(x + w / 2 + 0.01, pp_pct[order], w, color=ORANGE, label="% of participants (\u22652 of 3 folds)", zorder=3)
ax.axhline(50, color=INK2, lw=0.8, ls=(0, (3, 2)))
ax.text(len(order) - 0.45, 51.5, "50%", ha="right", va="bottom", fontsize=6.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([CH[i] for i in order], fontsize=6.3, rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylim(0, 78); ax.set_yticks([0, 25, 50, 75]); ax.set_ylabel("Selected (%)")
ax.set_title("(B) Selection stability, 10-channel montage", fontsize=7.5, weight="bold", loc="left", color=INK)
ax.legend(frameon=False, fontsize=6.3, loc="upper right", ncol=1, handlelength=1.0, borderaxespad=0.1)
tidy(ax)

ax = fig.add_subplot(gs[2])
names = [p[0] for p in perm][::-1]; vals = [p[1] for p in perm][::-1]
cols = [ORANGE if n == "Eye" else BLUE for n in names]
ax.barh(names, vals, color=cols, height=0.6, zorder=3)
for i, v in enumerate(vals):
    ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=6.5, color=INK)
ax.set_xlim(0, 13.5); ax.set_xlabel("Accuracy drop (points)")
ax.set_title("(C) Permutation", fontsize=7.5, weight="bold", loc="left", color=INK)
tidy(ax, "x")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig3_channels.{ext}"), dpi=600, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------------------------ Figure 4
cm = np.array(R["confusion_rownorm_pct"])
pc = R["per_class_fusion"]
met = np.array([[pc[e]["precision"], pc[e]["recall"], pc[e]["specificity"], pc[e]["f1"]] for e in EMO])
lab = ["Disgust", "Fear", "Sadness", "Neutral", "Happiness"]

fig = plt.figure(figsize=(3.3, 2.05))
gs = fig.add_gridspec(1, 2, width_ratios=[5, 4], left=0.235, right=0.99, bottom=0.25, top=0.86, wspace=0.08)
cmap_b = LinearSegmentedColormap.from_list("b", SEQ_BLUE)
cmap_o = LinearSegmentedColormap.from_list("o", SEQ_ORANGE)

ax = fig.add_subplot(gs[0])
ax.imshow(cm, cmap=cmap_b, vmin=0, vmax=100, aspect="auto")
for i in range(5):
    for j in range(5):
        v = cm[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                color="white" if v > 45 else INK, weight="bold" if i == j else "normal")
ax.set_xticks(range(5)); ax.set_xticklabels(["D", "F", "S", "N", "H"])
ax.set_yticks(range(5)); ax.set_yticklabels(lab)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.set_title("(A) Confusion (%)", fontsize=7.5, weight="bold", loc="left", color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True)
ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)

ax = fig.add_subplot(gs[1])
ax.imshow(met, cmap=cmap_o, vmin=40, vmax=100, aspect="auto")
for i in range(5):
    for j in range(4):
        v = met[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if v > 85 else INK)
ax.set_xticks(range(4)); ax.set_xticklabels(["Prec", "Rec", "Spec", "F1"])
ax.set_yticks([]); ax.set_xlabel("Metric (%)")
ax.set_title("(B) One-vs-rest", fontsize=7.5, weight="bold", loc="left", color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True)
ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig4_confusion.{ext}"), dpi=600, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("saved fig3_channels / fig4_confusion")
