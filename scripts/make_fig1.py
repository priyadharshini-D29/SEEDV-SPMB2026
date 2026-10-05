import os
# -*- coding: utf-8 -*-
"""Revised Figure 1 (reviewer request R2.3): shows both normalization paths.
(A) subject-dependent inductive path with training-trial session statistics;
(B) cross-subject causal-online path: source prior -> normalize with past moments ->
    predict -> update. Drawn at final print size (6.5 in wide) so fonts stay >= 7 pt."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
W, H = 6.5, 2.75
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

BLUE, ORANGE, GREY, PURPLE, GREEN, RED = "#2b5d8a", "#b8742a", "#59626e", "#5b4a9b", "#2e7a45", "#b03a2e"
FILL = {BLUE: "#dde8f3", ORANGE: "#fbe6cc", GREY: "#eef0f3", PURPLE: "#e9e3f5", GREEN: "#dcefe1", RED: "#f7e0dd"}


def box(x, y, w, h, title, sub, c, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=FILL[c], ec=c, lw=1.2))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs, weight="bold", color="#1b1f24")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.6, color="#2a2f36")


def arrow(x0, y0, x1, y1, c="#3d434b", style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8, lw=1.1,
                                 color=c, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


# ---- Panel A: subject-dependent, inductive ----
ax.text(0.06, 2.62, "(A) Subject-dependent, trial-independent evaluation (inductive normalization)",
        fontsize=7.8, weight="bold", va="center")
yA, hA = 1.62, 0.78
box(0.06, yA + 0.42, 0.95, 0.40, "EEG DE", "310 (62 ch \u00d7 5)", BLUE)
box(0.06, yA - 0.04, 0.95, 0.40, "Eye movement", "33 features", ORANGE)
box(1.25, yA - 0.04, 1.20, 0.86, "Session z-score", "\u03bc, \u03c3 from training\ntrials only, Eq. (1)", GREY)
box(2.70, yA + 0.42, 0.95, 0.40, "Channel select", "optional, EEG only", BLUE, fs=7.2)
box(3.88, yA + 0.17, 0.80, 0.42, "Concatenate", "343-d", PURPLE)
box(4.92, yA + 0.17, 0.80, 0.42, "Logistic reg.", "1,720 params", PURPLE)
box(5.86, yA + 0.17, 0.60, 0.42, "5 classes", "", GREEN)
arrow(1.01, yA + 0.62, 1.25, yA + 0.62, BLUE); arrow(1.01, yA + 0.16, 1.25, yA + 0.16, ORANGE)
arrow(2.45, yA + 0.62, 2.70, yA + 0.62, BLUE)
arrow(3.65, yA + 0.62, 3.88, yA + 0.48, BLUE)
arrow(2.45, yA + 0.16, 3.88, yA + 0.30, ORANGE)
arrow(4.68, yA + 0.38, 4.92, yA + 0.38); arrow(5.72, yA + 0.38, 5.86, yA + 0.38)

ax.plot([0.04, W - 0.04], [1.40, 1.40], color="#c3c8cf", lw=0.8, ls=(0, (3, 2)))

# ---- Panel B: cross-subject, causal-online ----
ax.text(0.06, 1.26, "(B) Unseen-subject (LOSO) evaluation: causal-online normalization, reset at each session",
        fontsize=7.8, weight="bold", va="center")
yB, hB = 0.36, 0.60          # boxes span y = 0.36 .. 0.96
box(0.06, yB, 1.10, hB, "Source prior", "\u03bc\u2080, \u03c3\u2080\u00b2 from 15 training\nsubjects; n\u2080 = 20", GREY)
box(1.38, yB, 1.12, hB, "Running moments", "\u03bc\u209c, \u03c3\u209c\u00b2 from prior and\nx\u2081 \u2026 x\u209c\u208b\u2081, Eq. (2)", GREY, fs=7.2)
box(2.72, yB, 1.12, hB, "Normalize x\u209c", "z\u209c = (x\u209c \u2212 \u03bc\u209c) / (\u03c3\u209c + \u03b5)\nEq. (3)", BLUE)
box(4.06, yB, 1.06, hB, "Predict \u0177\u209c", "frozen logistic\nregression", PURPLE)
box(5.34, yB, 1.12, hB, "Then update", "add x\u209c to the\nrunning sums", RED)
arrow(1.16, yB + hB / 2, 1.38, yB + hB / 2, GREY)
arrow(2.50, yB + hB / 2, 2.72, yB + hB / 2, GREY)
arrow(3.84, yB + hB / 2, 4.06, yB + hB / 2)
arrow(5.12, yB + hB / 2, 5.34, yB + hB / 2)
# unlabeled target segment enters from above
ax.text(3.28, 1.115, "target segment x\u209c (unlabeled)", fontsize=6.6, color=ORANGE, ha="center", va="center",
        weight="bold")
arrow(3.28, 1.06, 3.28, yB + hB, ORANGE)
# feedback loop below the boxes
ax.plot([5.90, 5.90], [yB, 0.17], color=RED, lw=1.1)
ax.plot([5.90, 1.94], [0.17, 0.17], color=RED, lw=1.1)
arrow(1.94, 0.17, 1.94, yB, RED)
ax.text(3.92, 0.06, "state passed to t + 1 only after \u0177\u209c is emitted", fontsize=6.6, color=RED,
        ha="center", va="center", style="italic")

fig.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "fig1_pipeline.png"), dpi=600)
fig.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "fig1_pipeline.pdf"))
print("saved")
