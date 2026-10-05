# -*- coding: utf-8 -*-
"""Figure 2 (full width), redrawn from the audited LOSO run (results/workstation/loso_final_arrays.npz, RESULTS_loso_final.md).
(A) accuracy by normalization access on the common last-50% window (mean, SD across participants);
(B) causal-online accuracy by session decile; (C) per-participant accuracy for the three main conditions.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "results", "workstation", "loso_final_arrays.npz"))
ind, on, tr = d["inductive"] * 100, d["online"] * 100, d["transductive"] * 100
dec, subs = d["decile"], d["subs"]

BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY, LIGHT = "#6f6e69", "#b9b8b2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


fig = plt.figure(figsize=(6.5, 3.05))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], width_ratios=[1, 1.2], left=0.07, right=0.99,
                      bottom=0.1, top=0.93, hspace=0.62, wspace=0.2)

# (A) access bars; values and SDs from RESULTS_loso_final.md (common last-50% window)
ax = fig.add_subplot(gs[0, 0])
labels = ["Inductive", "Calib 20%\n(chrono)", "Calib 20%\n(repr.)", "Causal\nonline", "Transductive"]
vals = [43.19, 47.78, 64.77, 73.59, 74.30]
sds = [9.7, 7.8, 9.4, 9.4, 8.2]
assert abs(vals[0] - ind.mean()) < 0.01 and abs(vals[3] - on.mean()) < 0.01 and abs(vals[4] - tr.mean()) < 0.01
assert abs(sds[0] - ind.std()) < 0.06 and abs(sds[3] - on.std()) < 0.06 and abs(sds[4] - tr.std()) < 0.06
ax.bar(range(5), vals, 0.62, yerr=sds, color=[GREY, LIGHT, LIGHT, BLUE, ORANGE], zorder=3,
       error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK})
for i, v in enumerate(vals):
    ax.text(i, v + sds[i] + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=6.3)
ax.set_ylim(0, 98); ax.set_yticks([0, 25, 50, 75]); ax.set_ylabel("Accuracy (%)")
ax.set_title("(A) Accuracy by normalization access", fontsize=7.5, weight="bold", loc="left", color=INK)
tidy(ax)

# (B) causal-online accuracy by session decile
ax = fig.add_subplot(gs[0, 1])
ax.axhline(74.3, color=ORANGE, lw=0.9, ls=(0, (3, 2)))
ax.axhline(43.2, color=GREY, lw=0.9, ls=(0, (3, 2)))
ax.plot(range(10), dec, color=BLUE, lw=2, marker="o", ms=4, mec="white", mew=0.8, zorder=3)
ax.text(2.5, 75.3, "transductive, 74.3", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.text(9.45, 44.9, "inductive, 43.2", ha="right", va="bottom", fontsize=6.5, color=INK2)
ax.set_xticks(range(10)); ax.set_xticklabels([f"{i * 10}\u2013{(i + 1) * 10}" for i in range(10)], fontsize=5.6)
ax.set_xlim(-0.5, 9.5); ax.set_ylim(38, 90); ax.set_yticks([40, 50, 60, 70, 80])
ax.set_xlabel("Session progress (%)", labelpad=1.5); ax.set_ylabel("Accuracy (%)")
ax.set_title("(B) Causal-online accuracy over a session", fontsize=7.5, weight="bold", loc="left", color=INK)
tidy(ax)

# (C) per participant, ordered by causal-online accuracy
ax = fig.add_subplot(gs[1, :])
order = np.argsort(on); x = np.arange(len(subs)); w = 0.26
ax.bar(x - w - 0.01, ind[order], w, color=GREY, label="Inductive", zorder=3)
ax.bar(x, on[order], w, color=BLUE, label="Causal online", zorder=3)
ax.bar(x + w + 0.01, tr[order], w, color=ORANGE, label="Transductive", zorder=3)
ax.set_xticks(x); ax.set_xticklabels([f"S{subs[i]:02d}" for i in order], fontsize=6.3)
ax.set_xlim(-0.6, len(subs) - 0.4); ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Accuracy (%)")
ax.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=3, handlelength=1.0, borderaxespad=0.1, columnspacing=1.2)
ax.set_title("(C) Per participant (ordered by causal-online accuracy)", fontsize=7.5, weight="bold", loc="left", color=INK)
tidy(ax)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig2_revised.{ext}"), dpi=600, bbox_inches="tight", pad_inches=0.02)
from PIL import Image
print("saved fig2_revised", Image.open(os.path.join(ROOT, "figures", "fig2_revised.png")).size)
