#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leakage-safe accuracy improvement study for the SEED-V SPMB paper.

This script preserves the verified outer evaluation:
  * subject-dependent, trial-independent 3-fold evaluation;
  * trials are never divided between training and testing;
  * per-session normalization uses outer-training trials only.

It evaluates three models:
  1. Baseline early-fusion Logistic Regression (C=1);
  2. Early fusion with C selected by grouped inner cross-validation;
  3. Probability-level late fusion, with EEG C, eye C, and the fusion weight
     selected using grouped inner cross-validation.

Every method is reported at:
  * segment level; and
  * trial level, by averaging class probabilities across segments belonging
    to the same held-out trial.

Trial-level results are a separate endpoint and must not be presented as
segment-level accuracy.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import scipy
import sklearn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

SEED = 0
SUBJECTS = tuple(range(1, 17))
EMOTIONS = ("Disgust", "Fear", "Sad", "Neutral", "Happy")
N_CLASSES = len(EMOTIONS)
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
ALPHA_GRID = tuple(np.linspace(0.0, 1.0, 11))

METHODS = (
    "baseline_early",
    "tuned_early",
    "tuned_late",
)
METHOD_NAMES = {
    "baseline_early": "Early fusion, C=1",
    "tuned_early": "Nested-tuned early fusion",
    "tuned_late": "Nested-tuned late fusion",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run nested leakage-safe tuning and trial aggregation for SEED-V."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("SEEDV_DATA", here.parent / "data")),
        help="Directory containing EEG_DE_features and Eye_movement_features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here.parent / "results" / "cloud_instance" / "spmb_boost_results",
        help="Directory for new boost-study result files.",
    )
    return parser.parse_args()


def validate_data_root(data_root: Path) -> None:
    missing = []
    for subject in SUBJECTS:
        for folder in ("EEG_DE_features", "Eye_movement_features"):
            path = data_root / folder / f"{subject}_123.npz"
            if not path.is_file():
                missing.append(path)
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:8])
        suffix = "\n  ..." if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} expected data file(s):\n{preview}{suffix}"
        )


def load_subject(data_root: Path, subject: int):
    eeg_npz = np.load(
        data_root / "EEG_DE_features" / f"{subject}_123.npz",
        allow_pickle=True,
    )
    eye_npz = np.load(
        data_root / "Eye_movement_features" / f"{subject}_123.npz",
        allow_pickle=True,
    )
    eeg_trials = pickle.loads(eeg_npz["data"].tobytes())
    labels = pickle.loads(eeg_npz["label"].tobytes())
    eye_trials = pickle.loads(eye_npz["data"].tobytes())

    eeg_rows, eye_rows, y_rows, group_rows, session_rows = [], [], [], [], []
    trials = sorted(
        set(eeg_trials.keys()) & set(eye_trials.keys()) & set(labels.keys())
    )
    if not trials:
        raise ValueError(f"Subject {subject}: no aligned trials were found.")

    for trial in trials:
        eeg = np.asarray(eeg_trials[trial], dtype=np.float32)
        eye = np.asarray(eye_trials[trial], dtype=np.float32)
        y = np.asarray(labels[trial]).astype(int).ravel()
        n = min(eeg.shape[0], eye.shape[0], y.shape[0])
        if n == 0:
            continue
        eeg_rows.append(eeg[:n])
        eye_rows.append(eye[:n])
        y_rows.append(y[:n])
        group_rows.append(np.full(n, trial, dtype=int))
        session_rows.append(np.full(n, trial // 15, dtype=int))

    return (
        np.concatenate(eeg_rows),
        np.concatenate(eye_rows),
        np.concatenate(y_rows),
        np.concatenate(group_rows),
        np.concatenate(session_rows),
    )


def training_only_normalize(
    eeg: np.ndarray,
    eye: np.ndarray,
    sessions: np.ndarray,
    training_indices: np.ndarray,
):
    """Normalize every row from statistics estimated only on training indices."""
    train_mask = np.zeros(len(sessions), dtype=bool)
    train_mask[training_indices] = True
    eeg_norm = eeg.copy()
    eye_norm = eye.copy()

    for session in np.unique(sessions):
        destination = sessions == session
        source = destination & train_mask
        if source.sum() < 2:
            raise RuntimeError(
                f"Session {session} has fewer than two training samples. "
                "No transductive fallback is allowed."
            )
        for matrix in (eeg_norm, eye_norm):
            mean = np.nanmean(matrix[source], axis=0)
            std = np.nanstd(matrix[source], axis=0)
            matrix[destination] = np.nan_to_num(
                (matrix[destination] - mean) / (std + 1e-6),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

    return eeg_norm.astype(np.float32), eye_norm.astype(np.float32)


def fit_logreg(x: np.ndarray, y: np.ndarray, c_value: float):
    model = LogisticRegression(C=c_value, max_iter=2000)
    return model.fit(x, y)


def aligned_probabilities(model, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    aligned = np.zeros((len(x), N_CLASSES), dtype=float)
    aligned[:, model.classes_.astype(int)] = raw
    return aligned


def feature_view(modality: str, eeg: np.ndarray, eye: np.ndarray):
    if modality == "eeg":
        return eeg
    if modality == "eye":
        return eye
    if modality == "early":
        return np.concatenate((eeg, eye), axis=1).astype(np.float32)
    raise ValueError(f"Unknown modality: {modality}")


def inner_fold_views(
    eeg: np.ndarray,
    eye: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sessions: np.ndarray,
    outer_training: np.ndarray,
) -> list[dict]:
    """Create inner folds with normalization refitted on each inner training set."""
    splitter = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=SEED
    )
    folds = []
    for inner_train_local, inner_valid_local in splitter.split(
        eeg[outer_training],
        y[outer_training],
        groups[outer_training],
    ):
        inner_train = outer_training[inner_train_local]
        inner_valid = outer_training[inner_valid_local]
        eeg_norm, eye_norm = training_only_normalize(
            eeg, eye, sessions, inner_train
        )
        folds.append(
            {
                "y_train": y[inner_train],
                "y_valid": y[inner_valid],
                "eeg_train": eeg_norm[inner_train],
                "eeg_valid": eeg_norm[inner_valid],
                "eye_train": eye_norm[inner_train],
                "eye_valid": eye_norm[inner_valid],
                "early_train": np.concatenate(
                    (eeg_norm[inner_train], eye_norm[inner_train]), axis=1
                ),
                "early_valid": np.concatenate(
                    (eeg_norm[inner_valid], eye_norm[inner_valid]), axis=1
                ),
            }
        )
    return folds


def choose_c(folds: list[dict], modality: str) -> tuple[float, float]:
    scores = {}
    for c_value in C_GRID:
        fold_scores = []
        for fold in folds:
            model = fit_logreg(
                fold[f"{modality}_train"], fold["y_train"], c_value
            )
            prediction = model.predict(fold[f"{modality}_valid"])
            fold_scores.append(
                balanced_accuracy_score(fold["y_valid"], prediction)
            )
        scores[c_value] = float(np.mean(fold_scores))

    # Prefer a value nearer C=1 if numerical ties occur.
    best_c = max(
        C_GRID,
        key=lambda c: (round(scores[c], 12), -abs(np.log10(c))),
    )
    return float(best_c), scores[best_c]


def choose_late_weight(
    folds: list[dict], eeg_c: float, eye_c: float
) -> tuple[float, float]:
    fold_probabilities = []
    for fold in folds:
        eeg_model = fit_logreg(fold["eeg_train"], fold["y_train"], eeg_c)
        eye_model = fit_logreg(fold["eye_train"], fold["y_train"], eye_c)
        fold_probabilities.append(
            (
                fold["y_valid"],
                aligned_probabilities(eeg_model, fold["eeg_valid"]),
                aligned_probabilities(eye_model, fold["eye_valid"]),
            )
        )

    alpha_scores = {}
    for alpha in ALPHA_GRID:
        fold_scores = []
        for y_valid, eeg_probability, eye_probability in fold_probabilities:
            fused = alpha * eeg_probability + (1.0 - alpha) * eye_probability
            fold_scores.append(
                balanced_accuracy_score(y_valid, np.argmax(fused, axis=1))
            )
        alpha_scores[alpha] = float(np.mean(fold_scores))

    # Prefer alpha=0.5 if numerical ties occur.
    best_alpha = max(
        ALPHA_GRID,
        key=lambda alpha: (
            round(alpha_scores[alpha], 12),
            -abs(alpha - 0.5),
        ),
    )
    return float(best_alpha), alpha_scores[best_alpha]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def aggregate_trials(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
):
    trial_true, trial_pred, trial_ids = [], [], []
    for trial in np.unique(groups):
        mask = groups == trial
        labels = np.unique(y_true[mask])
        if len(labels) != 1:
            raise RuntimeError(
                f"Trial {trial} contains multiple labels: {labels.tolist()}."
            )
        mean_probability = probabilities[mask].mean(axis=0)
        trial_true.append(int(labels[0]))
        trial_pred.append(int(np.argmax(mean_probability)))
        trial_ids.append(int(trial))
    return (
        np.asarray(trial_true, dtype=int),
        np.asarray(trial_pred, dtype=int),
        np.asarray(trial_ids, dtype=int),
    )


def mean_ci95(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    half_width = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return float(values.mean() - half_width), float(values.mean() + half_width)


def cohen_dz(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    std = delta.std(ddof=1)
    if std == 0:
        return 0.0 if delta.mean() == 0 else float(np.sign(delta.mean()) * np.inf)
    return float(delta.mean() / std)


def paired_test(
    subject_scores: dict[tuple[str, str], list[float]],
    endpoint: str,
    first: str,
    second: str,
) -> dict:
    a = np.asarray(subject_scores[(endpoint, first)], dtype=float)
    b = np.asarray(subject_scores[(endpoint, second)], dtype=float)
    delta = a - b
    if np.allclose(delta, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(
            delta,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    low, high = mean_ci95(delta)
    tolerance = 1e-12
    return {
        "endpoint": endpoint,
        "comparison": f"{METHOD_NAMES[first]} vs {METHOD_NAMES[second]}",
        "n_subjects": len(delta),
        "mean_delta_accuracy": float(delta.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "cohen_dz": cohen_dz(delta),
        "wilcoxon_statistic": statistic,
        "p_raw": p_value,
        "wins": int(np.sum(delta > tolerance)),
        "ties": int(np.sum(np.abs(delta) <= tolerance)),
        "losses": int(np.sum(delta < -tolerance)),
    }


def holm_adjust(raw_p_values: list[float]) -> list[float]:
    p = np.asarray(raw_p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running_max = 0.0
    total = len(p)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * p[original_index])
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(p)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted.tolist()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_data_root(data_root)
    np.random.seed(SEED)

    print(f"Data root: {data_root}", flush=True)
    print(f"Output:    {output_dir}", flush=True)
    print("Outer CV:  trial-grouped, training-only normalization", flush=True)
    print("Inner CV:  trial-grouped, normalization refitted per inner fold", flush=True)

    pooled_truth = {
        (endpoint, method): []
        for endpoint in ("segment", "trial")
        for method in METHODS
    }
    pooled_prediction = {
        (endpoint, method): []
        for endpoint in ("segment", "trial")
        for method in METHODS
    }
    subject_scores = {
        (endpoint, method): []
        for endpoint in ("segment", "trial")
        for method in METHODS
    }
    subject_rows = []
    hyperparameter_rows = []
    trial_prediction_rows = []

    for subject in SUBJECTS:
        eeg, eye, y, groups, sessions = load_subject(data_root, subject)
        segment_probability = {
            method: np.full((len(y), N_CLASSES), np.nan, dtype=float)
            for method in METHODS
        }
        subject_trial_truth = {method: [] for method in METHODS}
        subject_trial_prediction = {method: [] for method in METHODS}

        outer_splitter = StratifiedGroupKFold(
            n_splits=3, shuffle=True, random_state=SEED
        )
        for fold_number, (outer_train, outer_test) in enumerate(
            outer_splitter.split(eeg, y, groups), start=1
        ):
            inner_folds = inner_fold_views(
                eeg, eye, y, groups, sessions, outer_train
            )
            early_c, early_inner_score = choose_c(inner_folds, "early")
            eeg_c, eeg_inner_score = choose_c(inner_folds, "eeg")
            eye_c, eye_inner_score = choose_c(inner_folds, "eye")
            alpha, late_inner_score = choose_late_weight(
                inner_folds, eeg_c, eye_c
            )

            eeg_norm, eye_norm = training_only_normalize(
                eeg, eye, sessions, outer_train
            )
            early_x = feature_view("early", eeg_norm, eye_norm)

            baseline_model = fit_logreg(early_x[outer_train], y[outer_train], 1.0)
            tuned_early_model = fit_logreg(
                early_x[outer_train], y[outer_train], early_c
            )
            eeg_model = fit_logreg(eeg_norm[outer_train], y[outer_train], eeg_c)
            eye_model = fit_logreg(eye_norm[outer_train], y[outer_train], eye_c)

            probabilities = {
                "baseline_early": aligned_probabilities(
                    baseline_model, early_x[outer_test]
                ),
                "tuned_early": aligned_probabilities(
                    tuned_early_model, early_x[outer_test]
                ),
            }
            eeg_probability = aligned_probabilities(
                eeg_model, eeg_norm[outer_test]
            )
            eye_probability = aligned_probabilities(
                eye_model, eye_norm[outer_test]
            )
            probabilities["tuned_late"] = (
                alpha * eeg_probability + (1.0 - alpha) * eye_probability
            )

            hyperparameter_rows.append(
                {
                    "subject": subject,
                    "outer_fold": fold_number,
                    "early_c": early_c,
                    "early_inner_balanced_accuracy": early_inner_score,
                    "eeg_c": eeg_c,
                    "eeg_inner_balanced_accuracy": eeg_inner_score,
                    "eye_c": eye_c,
                    "eye_inner_balanced_accuracy": eye_inner_score,
                    "late_eeg_weight_alpha": alpha,
                    "late_inner_balanced_accuracy": late_inner_score,
                }
            )

            for method in METHODS:
                segment_probability[method][outer_test] = probabilities[method]
                trial_true, trial_pred, trial_ids = aggregate_trials(
                    probabilities[method],
                    y[outer_test],
                    groups[outer_test],
                )
                subject_trial_truth[method].extend(trial_true.tolist())
                subject_trial_prediction[method].extend(trial_pred.tolist())
                for trial_id, true_label, predicted_label in zip(
                    trial_ids, trial_true, trial_pred
                ):
                    trial_prediction_rows.append(
                        {
                            "subject": subject,
                            "outer_fold": fold_number,
                            "trial": trial_id,
                            "method": method,
                            "true_label": true_label,
                            "true_emotion": EMOTIONS[true_label],
                            "predicted_label": predicted_label,
                            "predicted_emotion": EMOTIONS[predicted_label],
                            "correct": int(true_label == predicted_label),
                        }
                    )

        progress_parts = []
        for method in METHODS:
            if np.isnan(segment_probability[method]).any():
                raise RuntimeError(
                    f"Subject {subject}, {method}: incomplete outer predictions."
                )
            segment_pred = np.argmax(segment_probability[method], axis=1)
            trial_true = np.asarray(subject_trial_truth[method], dtype=int)
            trial_pred = np.asarray(subject_trial_prediction[method], dtype=int)

            segment_metrics = metrics(y, segment_pred)
            trial_metrics = metrics(trial_true, trial_pred)
            subject_scores[("segment", method)].append(
                segment_metrics["accuracy"]
            )
            subject_scores[("trial", method)].append(trial_metrics["accuracy"])
            pooled_truth[("segment", method)].append(y)
            pooled_prediction[("segment", method)].append(segment_pred)
            pooled_truth[("trial", method)].append(trial_true)
            pooled_prediction[("trial", method)].append(trial_pred)

            for endpoint, endpoint_metrics, n_items in (
                ("segment", segment_metrics, len(y)),
                ("trial", trial_metrics, len(trial_true)),
            ):
                subject_rows.append(
                    {
                        "subject": subject,
                        "endpoint": endpoint,
                        "method": method,
                        "n_items": n_items,
                        **endpoint_metrics,
                    }
                )
            progress_parts.append(
                f"{method}: seg {100 * segment_metrics['accuracy']:.1f}%/"
                f"trial {100 * trial_metrics['accuracy']:.1f}%"
            )
        print(
            f"Subject {subject:02d}/16 | " + " | ".join(progress_parts),
            flush=True,
        )

    summary_rows = []
    pooled_arrays = {}
    for endpoint in ("segment", "trial"):
        for method in METHODS:
            y_true = np.concatenate(pooled_truth[(endpoint, method)])
            y_pred = np.concatenate(pooled_prediction[(endpoint, method)])
            pooled_arrays[(endpoint, method)] = (y_true, y_pred)
            result = metrics(y_true, y_pred)
            scores = np.asarray(subject_scores[(endpoint, method)])
            ci_low, ci_high = mean_ci95(scores)
            if method in ("baseline_early", "tuned_early"):
                parameters = 1720
            else:
                parameters = 1725
            summary_rows.append(
                {
                    "endpoint": endpoint,
                    "method": method,
                    "method_name": METHOD_NAMES[method],
                    "n_items": len(y_true),
                    "parameters": parameters,
                    **result,
                    "subject_accuracy_mean": float(scores.mean()),
                    "subject_accuracy_std": float(scores.std(ddof=1)),
                    "subject_accuracy_ci95_low": ci_low,
                    "subject_accuracy_ci95_high": ci_high,
                }
            )

    tests = [
        paired_test(
            subject_scores, "segment", "tuned_early", "baseline_early"
        ),
        paired_test(
            subject_scores, "segment", "tuned_late", "baseline_early"
        ),
        paired_test(subject_scores, "trial", "tuned_early", "baseline_early"),
        paired_test(subject_scores, "trial", "tuned_late", "baseline_early"),
        paired_test(subject_scores, "trial", "tuned_late", "tuned_early"),
    ]
    for row, adjusted in zip(
        tests, holm_adjust([row["p_raw"] for row in tests])
    ):
        row["p_holm"] = adjusted

    write_csv(output_dir / "spmb_boost_summary.csv", summary_rows)
    write_csv(output_dir / "spmb_boost_subject_metrics.csv", subject_rows)
    write_csv(
        output_dir / "spmb_boost_hyperparameters.csv", hyperparameter_rows
    )
    write_csv(
        output_dir / "spmb_boost_trial_predictions.csv",
        trial_prediction_rows,
    )
    write_csv(output_dir / "spmb_boost_statistical_tests.csv", tests)

    trial_summaries = [
        row for row in summary_rows if row["endpoint"] == "trial"
    ]
    best_trial = max(trial_summaries, key=lambda row: row["accuracy"])
    best_key = ("trial", best_trial["method"])
    best_true, best_pred = pooled_arrays[best_key]
    best_cm = confusion_matrix(
        best_true, best_pred, labels=np.arange(N_CLASSES)
    )
    best_cm_pct = np.divide(
        best_cm,
        best_cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(best_cm, dtype=float),
        where=best_cm.sum(axis=1, keepdims=True) != 0,
    ) * 100.0
    cm_rows = []
    for index, emotion in enumerate(EMOTIONS):
        row = {"true_class": emotion}
        row.update(
            {
                predicted: float(best_cm_pct[index, pred_index])
                for pred_index, predicted in enumerate(EMOTIONS)
            }
        )
        cm_rows.append(row)
    write_csv(
        output_dir / "spmb_boost_best_trial_confusion_rownorm_pct.csv",
        cm_rows,
    )

    def pct(value: float) -> str:
        return f"{100 * value:.2f}"

    lines = [
        "# SPMB Leakage-Safe Inductive Boost Study",
        "",
        "## Evaluation safeguards",
        "",
        "- Outer test trials are completely held out.",
        "- Session normalization uses outer-training trials only.",
        "- Hyperparameters are selected by grouped inner cross-validation.",
        "- Inner-fold normalization is recomputed from inner-training trials only.",
        "- Trial accuracy averages class probabilities within each held-out trial.",
        "- Segment-level and trial-level results are reported separately.",
        "",
        "## Results",
        "",
        "| Endpoint | Method | Parameters | Accuracy (%) | Balanced acc. (%) | Macro-F1 | MCC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['endpoint'].title()} | {row['method_name']} | "
            f"{row['parameters']} | {pct(row['accuracy'])} | "
            f"{pct(row['balanced_accuracy'])} | {row['macro_f1']:.3f} | "
            f"{row['mcc']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Paired subject-level comparisons",
            "",
            "| Endpoint | Comparison | Mean Δ (pp) | 95% CI (pp) | Cohen dz | "
            "Raw p | Holm p | W/T/L |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tests:
        dz = (
            f"{row['cohen_dz']:.3f}"
            if np.isfinite(row["cohen_dz"])
            else str(row["cohen_dz"])
        )
        lines.append(
            f"| {row['endpoint'].title()} | {row['comparison']} | "
            f"{100 * row['mean_delta_accuracy']:+.2f} | "
            f"[{100 * row['ci95_low']:+.2f}, "
            f"{100 * row['ci95_high']:+.2f}] | {dz} | "
            f"{row['p_raw']:.4g} | {row['p_holm']:.4g} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )

    baseline_segment = next(
        row
        for row in summary_rows
        if row["endpoint"] == "segment"
        and row["method"] == "baseline_early"
    )
    difference_pp = 100 * (baseline_segment["accuracy"] - 0.697)
    selected_cs = [row["early_c"] for row in hyperparameter_rows]
    selected_alphas = [
        row["late_eeg_weight_alpha"] for row in hyperparameter_rows
    ]

    lines.extend(
        [
            "",
            "## Validation and interpretation",
            "",
            f"- Baseline segment accuracy: {pct(baseline_segment['accuracy'])}%.",
            f"- Difference from the manuscript's rounded 69.7%: "
            f"{difference_pp:+.2f} percentage points.",
            f"- Best trial-level method: {best_trial['method_name']}.",
            f"- Best leakage-safe trial-level accuracy: "
            f"{pct(best_trial['accuracy'])}%.",
            f"- Median selected early-fusion C: "
            f"{float(np.median(selected_cs)):g}.",
            f"- Median selected late-fusion EEG weight: "
            f"{float(np.median(selected_alphas)):.2f}.",
        ]
    )
    if abs(difference_pp) <= 0.30:
        lines.append(
            "- The inductive baseline reproduces within the predeclared "
            "±0.30 percentage-point tolerance."
        )
    else:
        lines.append(
            "- **Warning:** the baseline reproduction differs by more than "
            "0.30 percentage points."
        )
    if best_trial["accuracy"] >= 0.80:
        lines.append(
            "- The 80% threshold was crossed at the **trial level**. It must "
            "not be relabelled as segment-level accuracy."
        )
    else:
        lines.append(
            "- The 80% threshold was not crossed under this leakage-safe study."
        )

    lines.extend(
        [
            "",
            "## Best trial-level per-class recall",
            "",
        ]
    )
    for emotion, recall in zip(EMOTIONS, np.diag(best_cm_pct)):
        lines.append(f"- {emotion}: {recall:.1f}%")
    lines.extend(
        [
            "",
            "Software:",
            f"Python {sys.version.split()[0]}, NumPy {np.__version__}, "
            f"SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}.",
            "",
        ]
    )

    report = "\n".join(lines)
    (output_dir / "SPMB_INDUCTIVE_BOOST_RESULTS.md").write_text(
        report, encoding="utf-8"
    )
    print("\n" + report, flush=True)
    print(f"\nSaved new results to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
