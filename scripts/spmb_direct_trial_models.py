#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct trial-representation study for the SEED-V SPMB paper.

This is a final, leakage-safe attempt to improve trial-level emotion
classification by changing the representation rather than reusing another
segment classifier.

Safeguards
----------
* Outer test trials are completely held out.
* Per-session EEG and eye normalization uses outer-training trials only.
* Every inner-validation fold recomputes normalization from its own
  inner-training trials.
* Feature selection, scaling, PCA, model selection, and hyperparameter
  selection are fitted inside training folds.
* The primary "nested selector" chooses the representation/model family using
  inner-fold balanced accuracy, never outer-test accuracy.

The script compares:
* the verified segment-probability averaging reference;
* trial-mean Logistic Regression;
* mean/std SelectKBest Logistic Regression;
* richer-moment SelectKBest Logistic Regression;
* trial-mean PCA + RBF-SVM;
* trial-mean shrinkage LDA;
* richer-moment Extra Trees; and
* a nested selector choosing among those direct-trial families.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import scipy
import sklearn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SEED = 0
SUBJECTS = tuple(range(1, 17))
EMOTIONS = ("Disgust", "Fear", "Sad", "Neutral", "Happy")
N_CLASSES = len(EMOTIONS)

REFERENCE = "segment_probability_reference"
SELECTOR = "nested_selector"
FAMILIES = (
    "mean_logreg",
    "meanstd_select_logreg",
    "moments_select_logreg",
    "mean_pca_svm",
    "mean_shrinkage_lda",
    "moments_extra_trees",
)
METHODS = (REFERENCE,) + FAMILIES + (SELECTOR,)

METHOD_NAMES = {
    REFERENCE: "Segment-probability averaging reference",
    "mean_logreg": "Trial mean + Logistic Regression",
    "meanstd_select_logreg": "Trial mean/std + selected-feature LogReg",
    "moments_select_logreg": "Trial moments + selected-feature LogReg",
    "mean_pca_svm": "Trial mean + PCA-RBF-SVM",
    "mean_shrinkage_lda": "Trial mean + shrinkage LDA",
    "moments_extra_trees": "Trial moments + Extra Trees",
    SELECTOR: "Nested-selected direct-trial model",
}

FAMILY_REPRESENTATION = {
    "mean_logreg": "mean",
    "meanstd_select_logreg": "meanstd",
    "moments_select_logreg": "moments",
    "mean_pca_svm": "mean",
    "mean_shrinkage_lda": "mean",
    "moments_extra_trees": "moments",
}


def product_dict(**kwargs):
    keys = tuple(kwargs)
    for values in itertools.product(*(kwargs[key] for key in keys)):
        yield dict(zip(keys, values))


FAMILY_CONFIGS = {
    "mean_logreg": list(
        product_dict(C=(0.01, 0.1, 1.0, 10.0, 100.0))
    ),
    "meanstd_select_logreg": list(
        product_dict(k=(25, 50, 100, 200), C=(0.1, 1.0, 10.0))
    ),
    "moments_select_logreg": list(
        product_dict(k=(25, 50, 100, 200), C=(0.1, 1.0, 10.0))
    ),
    "mean_pca_svm": list(
        product_dict(n_components=(5, 10, 15), C=(0.1, 1.0, 10.0))
    ),
    "mean_shrinkage_lda": list(
        product_dict(shrinkage=("auto", 0.1, 0.5, 0.9))
    ),
    "moments_extra_trees": list(
        product_dict(max_features=("sqrt", 0.3), min_samples_leaf=(1, 2))
    ),
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate direct trial-level EEG-eye representations with nested CV."
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
        default=here.parent / "results" / "cloud_instance" / "spmb_trial_results",
        help="Directory for new direct-trial result files.",
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
                "No test-assisted fallback is allowed."
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


def group_ids(indices: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.unique(groups[indices])


def trial_representation(
    eeg: np.ndarray,
    eye: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    selected_groups: np.ndarray,
    representation: str,
):
    features, labels, trial_ids = [], [], []
    combined = np.concatenate((eeg, eye), axis=1)

    for trial in np.sort(selected_groups):
        mask = groups == trial
        trial_labels = np.unique(y[mask])
        if len(trial_labels) != 1:
            raise RuntimeError(
                f"Trial {trial} contains multiple labels: "
                f"{trial_labels.tolist()}."
            )
        block = combined[mask]
        mean = block.mean(axis=0)
        if representation == "mean":
            vector = mean
        else:
            std = block.std(axis=0)
            if representation == "meanstd":
                vector = np.concatenate((mean, std))
            elif representation == "moments":
                q25, q75 = np.percentile(block, (25, 75), axis=0)
                if len(block) > 1:
                    mean_absolute_change = np.abs(np.diff(block, axis=0)).mean(
                        axis=0
                    )
                else:
                    mean_absolute_change = np.zeros(block.shape[1])
                vector = np.concatenate(
                    (mean, std, q75 - q25, mean_absolute_change)
                )
            else:
                raise ValueError(f"Unknown representation: {representation}")
        features.append(
            np.nan_to_num(
                vector, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
        )
        labels.append(int(trial_labels[0]))
        trial_ids.append(int(trial))

    return (
        np.vstack(features),
        np.asarray(labels, dtype=int),
        np.asarray(trial_ids, dtype=int),
    )


def aligned_probabilities(model, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    aligned = np.zeros((len(x), N_CLASSES), dtype=float)
    aligned[:, model.classes_.astype(int)] = raw
    return aligned


def reference_trial_predictions(
    eeg: np.ndarray,
    eye: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
):
    x = np.concatenate((eeg, eye), axis=1)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(
        x[train_indices], y[train_indices]
    )
    probabilities = aligned_probabilities(model, x[test_indices])
    test_groups = groups[test_indices]
    y_test = y[test_indices]
    true, predicted, trials = [], [], []
    for trial in np.sort(np.unique(test_groups)):
        mask = test_groups == trial
        labels = np.unique(y_test[mask])
        if len(labels) != 1:
            raise RuntimeError(
                f"Trial {trial} contains multiple labels: {labels.tolist()}."
            )
        true.append(int(labels[0]))
        predicted.append(int(np.argmax(probabilities[mask].mean(axis=0))))
        trials.append(int(trial))
    return (
        np.asarray(true, dtype=int),
        np.asarray(predicted, dtype=int),
        np.asarray(trials, dtype=int),
    )


def build_model(family: str, config: dict):
    if family == "mean_logreg":
        return Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(C=config["C"], max_iter=3000),
                ),
            )
        )
    if family in ("meanstd_select_logreg", "moments_select_logreg"):
        return Pipeline(
            (
                ("select", SelectKBest(f_classif, k=config["k"])),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(C=config["C"], max_iter=3000),
                ),
            )
        )
    if family == "mean_pca_svm":
        return Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=config["n_components"],
                        svd_solver="full",
                    ),
                ),
                (
                    "classifier",
                    SVC(C=config["C"], gamma="scale", kernel="rbf"),
                ),
            )
        )
    if family == "mean_shrinkage_lda":
        return Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(
                        solver="lsqr", shrinkage=config["shrinkage"]
                    ),
                ),
            )
        )
    if family == "moments_extra_trees":
        return ExtraTreesClassifier(
            n_estimators=200,
            max_features=config["max_features"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown family: {family}")


def make_inner_contexts(
    eeg: np.ndarray,
    eye: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sessions: np.ndarray,
    outer_training: np.ndarray,
) -> list[dict]:
    contexts = []
    splitter = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=SEED
    )
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
        context = {}
        for representation in ("mean", "meanstd", "moments"):
            x_train, y_train, _ = trial_representation(
                eeg_norm,
                eye_norm,
                y,
                groups,
                group_ids(inner_train, groups),
                representation,
            )
            x_valid, y_valid, _ = trial_representation(
                eeg_norm,
                eye_norm,
                y,
                groups,
                group_ids(inner_valid, groups),
                representation,
            )
            context[representation] = (
                x_train,
                y_train,
                x_valid,
                y_valid,
            )
        contexts.append(context)
    return contexts


def tune_family(family: str, contexts: list[dict]):
    representation = FAMILY_REPRESENTATION[family]
    best_config, best_score = None, -np.inf
    for config in FAMILY_CONFIGS[family]:
        fold_scores = []
        for context in contexts:
            x_train, y_train, x_valid, y_valid = context[representation]
            model = build_model(family, config)
            model.fit(x_train, y_train)
            fold_scores.append(
                balanced_accuracy_score(y_valid, model.predict(x_valid))
            )
        score = float(np.mean(fold_scores))
        if score > best_score + 1e-12:
            best_config = dict(config)
            best_score = score
    return best_config, best_score


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


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
    low, high = mean_ci95(delta)
    tolerance = 1e-12
    return {
        "comparison": f"{METHOD_NAMES[first]} vs {METHOD_NAMES[second]}",
        "first": first,
        "second": second,
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
    print("Outer CV:  held-out trials, training-only normalization", flush=True)
    print("Inner CV:  normalization and model selection refitted per fold", flush=True)

    pooled_true = {method: [] for method in METHODS}
    pooled_pred = {method: [] for method in METHODS}
    subject_scores = {method: [] for method in METHODS}
    subject_rows = []
    fold_rows = []
    trial_rows = []

    for subject in SUBJECTS:
        eeg, eye, y, groups, sessions = load_subject(data_root, subject)
        subject_true = {method: [] for method in METHODS}
        subject_pred = {method: [] for method in METHODS}

        outer_splitter = StratifiedGroupKFold(
            n_splits=3, shuffle=True, random_state=SEED
        )
        for outer_fold, (outer_train, outer_test) in enumerate(
            outer_splitter.split(eeg, y, groups), start=1
        ):
            contexts = make_inner_contexts(
                eeg, eye, y, groups, sessions, outer_train
            )
            best_configs, inner_scores = {}, {}
            for family in FAMILIES:
                best_configs[family], inner_scores[family] = tune_family(
                    family, contexts
                )

            selected_family = max(
                FAMILIES,
                key=lambda family: (
                    round(inner_scores[family], 12),
                    -FAMILIES.index(family),
                ),
            )

            eeg_norm, eye_norm = training_only_normalize(
                eeg, eye, sessions, outer_train
            )
            reference_true, reference_pred, trial_ids = (
                reference_trial_predictions(
                    eeg_norm,
                    eye_norm,
                    y,
                    groups,
                    outer_train,
                    outer_test,
                )
            )
            fold_predictions = {REFERENCE: reference_pred}
            fold_truth = reference_true

            outer_representations = {}
            for representation in ("mean", "meanstd", "moments"):
                x_train, y_train, _ = trial_representation(
                    eeg_norm,
                    eye_norm,
                    y,
                    groups,
                    group_ids(outer_train, groups),
                    representation,
                )
                x_test, y_test, test_trial_ids = trial_representation(
                    eeg_norm,
                    eye_norm,
                    y,
                    groups,
                    group_ids(outer_test, groups),
                    representation,
                )
                if not np.array_equal(y_test, fold_truth):
                    raise RuntimeError("Trial-label ordering mismatch.")
                if not np.array_equal(test_trial_ids, trial_ids):
                    raise RuntimeError("Trial-ID ordering mismatch.")
                outer_representations[representation] = (
                    x_train,
                    y_train,
                    x_test,
                )

            for family in FAMILIES:
                representation = FAMILY_REPRESENTATION[family]
                x_train, y_train, x_test = outer_representations[
                    representation
                ]
                model = build_model(family, best_configs[family])
                model.fit(x_train, y_train)
                fold_predictions[family] = model.predict(x_test)

            fold_predictions[SELECTOR] = fold_predictions[selected_family].copy()

            for family in FAMILIES:
                fold_rows.append(
                    {
                        "subject": subject,
                        "outer_fold": outer_fold,
                        "family": family,
                        "representation": FAMILY_REPRESENTATION[family],
                        "inner_balanced_accuracy": inner_scores[family],
                        "selected_by_nested_selector": int(
                            family == selected_family
                        ),
                        "configuration": json.dumps(
                            best_configs[family], sort_keys=True
                        ),
                    }
                )

            for method in METHODS:
                subject_true[method].extend(fold_truth.tolist())
                subject_pred[method].extend(
                    np.asarray(fold_predictions[method], dtype=int).tolist()
                )
                for trial, true_label, predicted_label in zip(
                    trial_ids, fold_truth, fold_predictions[method]
                ):
                    trial_rows.append(
                        {
                            "subject": subject,
                            "outer_fold": outer_fold,
                            "trial": int(trial),
                            "method": method,
                            "true_label": int(true_label),
                            "true_emotion": EMOTIONS[int(true_label)],
                            "predicted_label": int(predicted_label),
                            "predicted_emotion": EMOTIONS[int(predicted_label)],
                            "correct": int(true_label == predicted_label),
                        }
                    )

        progress = []
        for method in METHODS:
            y_true = np.asarray(subject_true[method], dtype=int)
            y_pred = np.asarray(subject_pred[method], dtype=int)
            result = metric_row(y_true, y_pred)
            subject_scores[method].append(result["accuracy"])
            pooled_true[method].append(y_true)
            pooled_pred[method].append(y_pred)
            subject_rows.append(
                {
                    "subject": subject,
                    "method": method,
                    "n_trials": len(y_true),
                    **result,
                }
            )
            if method in (REFERENCE, SELECTOR):
                progress.append(
                    f"{method}: {100 * result['accuracy']:.1f}%"
                )
        print(
            f"Subject {subject:02d}/16 | " + " | ".join(progress),
            flush=True,
        )

    summary_rows = []
    pooled_arrays = {}
    for method in METHODS:
        y_true = np.concatenate(pooled_true[method])
        y_pred = np.concatenate(pooled_pred[method])
        pooled_arrays[method] = (y_true, y_pred)
        result = metric_row(y_true, y_pred)
        scores = np.asarray(subject_scores[method])
        ci_low, ci_high = mean_ci95(scores)
        summary_rows.append(
            {
                "method": method,
                "method_name": METHOD_NAMES[method],
                "n_trials": len(y_true),
                **result,
                "subject_accuracy_mean": float(scores.mean()),
                "subject_accuracy_std": float(scores.std(ddof=1)),
                "subject_accuracy_ci95_low": ci_low,
                "subject_accuracy_ci95_high": ci_high,
            }
        )

    tests = [
        paired_test(subject_scores, method, REFERENCE)
        for method in FAMILIES + (SELECTOR,)
    ]
    for row, adjusted in zip(
        tests, holm_adjust([row["p_raw"] for row in tests])
    ):
        row["p_holm"] = adjusted

    write_csv(output_dir / "spmb_trial_model_summary.csv", summary_rows)
    write_csv(output_dir / "spmb_trial_subject_metrics.csv", subject_rows)
    write_csv(output_dir / "spmb_trial_fold_selections.csv", fold_rows)
    write_csv(output_dir / "spmb_trial_predictions.csv", trial_rows)
    write_csv(output_dir / "spmb_trial_statistical_tests.csv", tests)

    selector_true, selector_pred = pooled_arrays[SELECTOR]
    selector_cm = confusion_matrix(
        selector_true, selector_pred, labels=np.arange(N_CLASSES)
    )
    selector_cm_pct = np.divide(
        selector_cm,
        selector_cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(selector_cm, dtype=float),
        where=selector_cm.sum(axis=1, keepdims=True) != 0,
    ) * 100.0
    cm_rows = []
    for index, emotion in enumerate(EMOTIONS):
        row = {"true_class": emotion}
        row.update(
            {
                predicted: float(selector_cm_pct[index, pred_index])
                for pred_index, predicted in enumerate(EMOTIONS)
            }
        )
        cm_rows.append(row)
    write_csv(
        output_dir / "spmb_nested_selector_confusion_rownorm_pct.csv",
        cm_rows,
    )

    selection_counts = Counter(
        row["family"]
        for row in fold_rows
        if row["selected_by_nested_selector"] == 1
    )
    summary_by_method = {row["method"]: row for row in summary_rows}
    reference_result = summary_by_method[REFERENCE]
    selector_result = summary_by_method[SELECTOR]

    def pct(value: float) -> str:
        return f"{100 * value:.2f}"

    lines = [
        "# SPMB Direct Trial-Representation Study",
        "",
        "## Evaluation safeguards",
        "",
        "- Complete outer test trials are held out.",
        "- EEG and eye normalization uses training trials only.",
        "- Inner-fold normalization is refitted using inner-training trials.",
        "- Scaling, feature selection and PCA occur inside training pipelines.",
        "- Hyperparameters and the primary model family are selected without "
        "outer-test labels.",
        "",
        "## Trial-level results",
        "",
        "| Method | Accuracy (%) | Balanced acc. (%) | Macro-F1 | MCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method_name']} | {pct(row['accuracy'])} | "
            f"{pct(row['balanced_accuracy'])} | "
            f"{row['macro_f1']:.3f} | {row['mcc']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Paired comparisons against the reference",
            "",
            "| Direct-trial method | Mean Δ (pp) | 95% CI (pp) | Cohen dz | "
            "Raw p | Holm p | W/T/L |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tests:
        dz = (
            f"{row['cohen_dz']:.3f}"
            if np.isfinite(row["cohen_dz"])
            else str(row["cohen_dz"])
        )
        lines.append(
            f"| {METHOD_NAMES[row['first']]} | "
            f"{100 * row['mean_delta_accuracy']:+.2f} | "
            f"[{100 * row['ci95_low']:+.2f}, "
            f"{100 * row['ci95_high']:+.2f}] | {dz} | "
            f"{row['p_raw']:.4g} | {row['p_holm']:.4g} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )

    lines.extend(
        [
            "",
            "## Nested selector choices",
            "",
        ]
    )
    for family in FAMILIES:
        lines.append(
            f"- {METHOD_NAMES[family]}: "
            f"{selection_counts.get(family, 0)} of 48 outer folds."
        )

    reference_difference = 100 * (reference_result["accuracy"] - 0.6764)
    lines.extend(
        [
            "",
            "## Validation and decision",
            "",
            f"- Reproduced probability-averaging reference: "
            f"{pct(reference_result['accuracy'])}%.",
            f"- Difference from the previous rounded 67.64%: "
            f"{reference_difference:+.2f} percentage points.",
            f"- Primary nested-selector accuracy: "
            f"{pct(selector_result['accuracy'])}%.",
        ]
    )
    if abs(reference_difference) <= 0.30:
        lines.append(
            "- The reference reproduces within the predeclared "
            "±0.30 percentage-point tolerance."
        )
    else:
        lines.append(
            "- **Warning:** the trial reference differs by more than "
            "0.30 percentage points."
        )
    if selector_result["accuracy"] >= 0.80:
        lines.append(
            "- The predeclared primary nested selector crossed 80% under the "
            "leakage-safe trial-level protocol."
        )
    else:
        lines.append(
            "- The predeclared primary nested selector did not cross 80%. "
            "Do not select another family post hoc as the proposed method."
        )

    lines.extend(
        [
            "",
            "## Nested-selector per-class recall",
            "",
        ]
    )
    for emotion, recall in zip(EMOTIONS, np.diag(selector_cm_pct)):
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
    (output_dir / "SPMB_DIRECT_TRIAL_RESULTS.md").write_text(
        report, encoding="utf-8"
    )
    print("\n" + report, flush=True)
    print(f"\nSaved new results to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
