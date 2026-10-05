#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPMB modality ablation for SEED-V.

Evaluates EEG-only, eye-movement-only, and EEG+eye early fusion under the
same subject-dependent, trial-independent protocol used by
trialindep_inductive.py:

  * 3-fold StratifiedGroupKFold, grouped by trial
  * training-trial-only normalization within each session
  * identical LogisticRegression classifier for every modality

The original scripts and result files are not modified. New outputs are
written to --output-dir (default: spmb_results).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import scipy
from scipy import stats
import sklearn
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
MODALITIES = ("EEG-only", "Eye-only", "EEG+Eye")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe SEED-V EEG/eye/fusion ablation for SPMB."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=here.parent / "data",
        help="Directory containing EEG_DE_features and Eye_movement_features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here.parent / "results" / "cloud_instance" / "spmb_results",
        help="Directory for new SPMB result files.",
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
        preview = "\n".join(f"  - {p}" for p in missing[:8])
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
    common_trials = sorted(
        set(eeg_trials.keys()) & set(eye_trials.keys()) & set(labels.keys())
    )
    if not common_trials:
        raise ValueError(f"Subject {subject}: no common EEG/eye/label trials.")

    for trial in common_trials:
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


def inductive_session_normalize(
    eeg: np.ndarray,
    eye: np.ndarray,
    sessions: np.ndarray,
    train_mask: np.ndarray,
):
    """Normalize all rows with statistics computed from training rows only."""
    eeg_norm = eeg.copy()
    eye_norm = eye.copy()

    for session in np.unique(sessions):
        all_rows = sessions == session
        source_rows = all_rows & train_mask
        if source_rows.sum() < 2:
            raise RuntimeError(
                f"Session {session} has fewer than two training samples. "
                "Refusing to fall back to test-assisted normalization."
            )

        for matrix in (eeg_norm, eye_norm):
            mean = np.nanmean(matrix[source_rows], axis=0)
            std = np.nanstd(matrix[source_rows], axis=0)
            matrix[all_rows] = np.nan_to_num(
                (matrix[all_rows] - mean) / (std + 1e-6),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

    return eeg_norm.astype(np.float32), eye_norm.astype(np.float32)


def feature_view(
    modality: str, eeg: np.ndarray, eye: np.ndarray
) -> np.ndarray:
    if modality == "EEG-only":
        return eeg
    if modality == "Eye-only":
        return eye
    if modality == "EEG+Eye":
        return np.concatenate((eeg, eye), axis=1).astype(np.float32)
    raise ValueError(f"Unknown modality: {modality}")


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def model_parameters(model: LogisticRegression) -> int:
    return int(model.coef_.size + model.intercept_.size)


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float(values.mean()), float(values.mean())
    sem = stats.sem(values)
    half_width = stats.t.ppf(0.975, values.size - 1) * sem
    return float(values.mean() - half_width), float(values.mean() + half_width)


def cohen_dz(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    std = delta.std(ddof=1)
    if std == 0:
        return 0.0 if delta.mean() == 0 else float(np.sign(delta.mean()) * np.inf)
    return float(delta.mean() / std)


def holm_adjust(raw_p_values: list[float]) -> list[float]:
    """Holm step-down family-wise correction."""
    p = np.asarray(raw_p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[original_index])
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(p)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted.tolist()


def paired_test(
    subject_scores: dict[str, list[float]], first: str, second: str
) -> dict:
    a = np.asarray(subject_scores[first], dtype=float)
    b = np.asarray(subject_scores[second], dtype=float)
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

    ci_low, ci_high = mean_ci95(delta)
    tolerance = 1e-12
    return {
        "comparison": f"{first} vs {second}",
        "first": first,
        "second": second,
        "n_subjects": len(delta),
        "mean_delta_accuracy": float(delta.mean()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohen_dz": cohen_dz(delta),
        "wilcoxon_statistic": statistic,
        "p_raw": p_value,
        "wins": int(np.sum(delta > tolerance)),
        "ties": int(np.sum(np.abs(delta) <= tolerance)),
        "losses": int(np.sum(delta < -tolerance)),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_data_root(data_root)

    np.random.seed(SEED)
    print(f"Data root: {data_root}", flush=True)
    print(f"Output:    {output_dir}", flush=True)
    print("Protocol:  3-fold trial-grouped, training-only session normalization", flush=True)

    fold_rows: list[dict] = []
    subject_rows: list[dict] = []
    subject_scores = {modality: [] for modality in MODALITIES}
    pooled_true = {modality: [] for modality in MODALITIES}
    pooled_pred = {modality: [] for modality in MODALITIES}
    efficiency = {
        modality: {
            "parameter_counts": [],
            "fit_seconds": [],
            "predict_seconds_per_sample": [],
        }
        for modality in MODALITIES
    }

    for subject in SUBJECTS:
        eeg, eye, y, groups, sessions = load_subject(data_root, subject)
        predictions = {
            modality: np.full(y.shape, -1, dtype=int) for modality in MODALITIES
        }
        splitter = StratifiedGroupKFold(
            n_splits=3, shuffle=True, random_state=SEED
        )

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(eeg, y, groups), start=1
        ):
            train_mask = np.zeros(y.size, dtype=bool)
            train_mask[train_idx] = True
            eeg_norm, eye_norm = inductive_session_normalize(
                eeg, eye, sessions, train_mask
            )

            for modality in MODALITIES:
                x = feature_view(modality, eeg_norm, eye_norm)
                model = LogisticRegression(max_iter=500)

                start = time.perf_counter()
                model.fit(x[train_idx], y[train_idx])
                fit_seconds = time.perf_counter() - start

                # One warm-up prediction, then an averaged timing measurement.
                model.predict(x[test_idx[: min(32, len(test_idx))]])
                repetitions = 10
                start = time.perf_counter()
                for _ in range(repetitions):
                    fold_prediction = model.predict(x[test_idx])
                predict_seconds = (time.perf_counter() - start) / repetitions

                predictions[modality][test_idx] = fold_prediction
                metrics = metric_row(y[test_idx], fold_prediction)
                parameters = model_parameters(model)
                per_sample = predict_seconds / len(test_idx)

                fold_rows.append(
                    {
                        "subject": subject,
                        "fold": fold,
                        "modality": modality,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_features": x.shape[1],
                        "parameters": parameters,
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "mcc": metrics["mcc"],
                        "fit_seconds": fit_seconds,
                        "predict_seconds_per_sample": per_sample,
                    }
                )
                efficiency[modality]["parameter_counts"].append(parameters)
                efficiency[modality]["fit_seconds"].append(fit_seconds)
                efficiency[modality]["predict_seconds_per_sample"].append(per_sample)

        for modality in MODALITIES:
            if np.any(predictions[modality] < 0):
                raise RuntimeError(
                    f"Subject {subject}, {modality}: some samples were not evaluated."
                )
            metrics = metric_row(y, predictions[modality])
            subject_scores[modality].append(metrics["accuracy"])
            pooled_true[modality].append(y)
            pooled_pred[modality].append(predictions[modality])
            subject_rows.append(
                {
                    "subject": subject,
                    "modality": modality,
                    "n_samples": len(y),
                    **metrics,
                }
            )

        progress = " | ".join(
            f"{m}: {100 * subject_scores[m][-1]:.1f}%" for m in MODALITIES
        )
        print(f"Subject {subject:02d}/16 | {progress}", flush=True)

    summary_rows = []
    pooled_arrays = {}
    for modality in MODALITIES:
        y_true = np.concatenate(pooled_true[modality])
        y_pred = np.concatenate(pooled_pred[modality])
        pooled_arrays[modality] = (y_true, y_pred)
        pooled_metrics = metric_row(y_true, y_pred)
        subject_accuracy = np.asarray(subject_scores[modality])
        ci_low, ci_high = mean_ci95(subject_accuracy)
        modality_folds = [r for r in fold_rows if r["modality"] == modality]
        summary_rows.append(
            {
                "modality": modality,
                "n_subjects": len(SUBJECTS),
                "n_samples": len(y_true),
                "n_features": modality_folds[0]["n_features"],
                "parameters": int(np.median(efficiency[modality]["parameter_counts"])),
                "pooled_accuracy": pooled_metrics["accuracy"],
                "pooled_balanced_accuracy": pooled_metrics["balanced_accuracy"],
                "pooled_macro_f1": pooled_metrics["macro_f1"],
                "pooled_mcc": pooled_metrics["mcc"],
                "subject_accuracy_mean": subject_accuracy.mean(),
                "subject_accuracy_std": subject_accuracy.std(ddof=1),
                "subject_accuracy_ci95_low": ci_low,
                "subject_accuracy_ci95_high": ci_high,
                "median_fit_seconds_per_fold": float(
                    np.median(efficiency[modality]["fit_seconds"])
                ),
                "median_prediction_microseconds_per_sample": float(
                    1e6
                    * np.median(
                        efficiency[modality]["predict_seconds_per_sample"]
                    )
                ),
            }
        )

    tests = [
        paired_test(subject_scores, "EEG+Eye", "EEG-only"),
        paired_test(subject_scores, "EEG+Eye", "Eye-only"),
        paired_test(subject_scores, "EEG-only", "Eye-only"),
    ]
    adjusted = holm_adjust([row["p_raw"] for row in tests])
    for row, p_adjusted in zip(tests, adjusted):
        row["p_holm"] = p_adjusted

    fusion_true, fusion_pred = pooled_arrays["EEG+Eye"]
    labels = np.arange(len(EMOTIONS))
    fusion_cm = confusion_matrix(fusion_true, fusion_pred, labels=labels)
    fusion_cm_pct = np.divide(
        fusion_cm,
        fusion_cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(fusion_cm, dtype=float),
        where=fusion_cm.sum(axis=1, keepdims=True) != 0,
    ) * 100.0

    write_csv(
        output_dir / "spmb_modality_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )
    write_csv(
        output_dir / "spmb_fold_metrics.csv",
        fold_rows,
        list(fold_rows[0].keys()),
    )
    write_csv(
        output_dir / "spmb_subject_metrics.csv",
        subject_rows,
        list(subject_rows[0].keys()),
    )
    write_csv(
        output_dir / "spmb_statistical_tests.csv",
        tests,
        list(tests[0].keys()),
    )

    cm_header = ["true_class"] + list(EMOTIONS)
    cm_count_rows = [
        {"true_class": EMOTIONS[i], **dict(zip(EMOTIONS, fusion_cm[i].tolist()))}
        for i in range(len(EMOTIONS))
    ]
    cm_pct_rows = [
        {
            "true_class": EMOTIONS[i],
            **dict(zip(EMOTIONS, np.round(fusion_cm_pct[i], 6).tolist())),
        }
        for i in range(len(EMOTIONS))
    ]
    write_csv(
        output_dir / "spmb_confusion_fusion_counts.csv", cm_count_rows, cm_header
    )
    write_csv(
        output_dir / "spmb_confusion_fusion_rownorm_pct.csv", cm_pct_rows, cm_header
    )

    efficiency_json = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "seed": SEED,
            "thread_limits": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        },
        "modalities": {
            row["modality"]: {
                "n_features": row["n_features"],
                "parameters": row["parameters"],
                "median_fit_seconds_per_fold": row[
                    "median_fit_seconds_per_fold"
                ],
                "median_prediction_microseconds_per_sample": row[
                    "median_prediction_microseconds_per_sample"
                ],
            }
            for row in summary_rows
        },
    }
    with (output_dir / "spmb_efficiency.json").open("w", encoding="utf-8") as handle:
        json.dump(efficiency_json, handle, indent=2)
        handle.write("\n")

    summary_by_name = {row["modality"]: row for row in summary_rows}
    fusion = summary_by_name["EEG+Eye"]
    recalls = np.diag(fusion_cm_pct)
    lines = [
        "# SPMB Modality Ablation — Leakage-Safe Trial-Independent Evaluation",
        "",
        "## Protocol",
        "",
        "- SEED-V subjects: 16",
        "- Split: 3-fold `StratifiedGroupKFold`, grouped by trial",
        "- Normalization: per-session statistics from training trials only",
        "- Classifier: identical logistic regression (`max_iter=500`) for all modalities",
        "- Seed: 0",
        "",
        "## Main results",
        "",
        "| Modality | Features | Parameters | Accuracy (%) | Balanced acc. (%) | Macro-F1 | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['modality']} | {row['n_features']} | {row['parameters']} | "
            f"{format_pct(row['pooled_accuracy'])} | "
            f"{format_pct(row['pooled_balanced_accuracy'])} | "
            f"{row['pooled_macro_f1']:.3f} | {row['pooled_mcc']:.3f} |"
        )

    lines.extend(
        [
            "",
            "Subject-level accuracy is summarized as mean ± sample SD and a 95% "
            "t-interval:",
            "",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"- {row['modality']}: "
            f"{format_pct(row['subject_accuracy_mean'])} ± "
            f"{format_pct(row['subject_accuracy_std'])}% "
            f"(95% CI {format_pct(row['subject_accuracy_ci95_low'])}–"
            f"{format_pct(row['subject_accuracy_ci95_high'])}%)."
        )

    lines.extend(
        [
            "",
            "## Paired subject-level comparisons",
            "",
            "| Comparison | Mean Δ accuracy (pp) | 95% CI (pp) | Cohen dz | "
            "Wilcoxon p | Holm p | W/T/L |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tests:
        dz = row["cohen_dz"]
        dz_text = f"{dz:.3f}" if np.isfinite(dz) else str(dz)
        lines.append(
            f"| {row['comparison']} | {100 * row['mean_delta_accuracy']:+.2f} | "
            f"[{100 * row['ci95_low']:+.2f}, {100 * row['ci95_high']:+.2f}] | "
            f"{dz_text} | {row['p_raw']:.4g} | {row['p_holm']:.4g} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )

    lines.extend(
        [
            "",
            "## Fusion per-class recall",
            "",
        ]
    )
    for emotion, recall in zip(EMOTIONS, recalls):
        lines.append(f"- {emotion}: {recall:.1f}%")

    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| Modality | Median fit/fold (s) | Median prediction (µs/sample) |",
            "|---|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['modality']} | "
            f"{row['median_fit_seconds_per_fold']:.4f} | "
            f"{row['median_prediction_microseconds_per_sample']:.3f} |"
        )

    expected_accuracy = 0.697
    difference_pp = 100.0 * (fusion["pooled_accuracy"] - expected_accuracy)
    lines.extend(
        [
            "",
            "## Reproduction check",
            "",
            f"- Fusion accuracy from this run: "
            f"{format_pct(fusion['pooled_accuracy'])}%.",
            f"- Difference from the manuscript's rounded 69.7% result: "
            f"{difference_pp:+.2f} percentage points.",
        ]
    )
    if abs(difference_pp) > 0.30:
        lines.append(
            "- **Warning:** the difference exceeds 0.30 percentage points. "
            "Check data identity and software-version effects before using the result."
        )
    else:
        lines.append(
            "- Reproduction is within the predeclared ±0.30 percentage-point tolerance."
        )

    lines.extend(
        [
            "",
            "All metrics above were generated by `spmb_modality_ablation.py`; "
            "the CSV and JSON files in this directory contain the underlying values.",
            "",
        ]
    )
    report = "\n".join(lines)
    (output_dir / "SPMB_MODALITY_RESULTS.md").write_text(report, encoding="utf-8")

    print("\n" + report, flush=True)
    print(f"\nSaved new files to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
