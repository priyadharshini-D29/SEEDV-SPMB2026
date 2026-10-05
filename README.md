# Leakage-Aware Causal Normalization for Lightweight Multimodal Emotion Recognition on SEED-V

Code and non-licensed results for the IEEE SPMB 2026 paper by Priyadharshini D and Shridevi S
(Centre for Neuroinformatics, Vellore Institute of Technology, Chennai).

The paper shows how much reported accuracy on SEED-V (EEG + eye movements, five emotions) depends on what the
evaluation pipeline is allowed to see, and specifies a temporally causal, label-free normalization for unseen
subjects.

| Split / endpoint | Normalization access | Accuracy (%) |
|---|---|---:|
| Segment-random | Whole session | 100.00 |
| Held-out trials | Whole session | 75.49 |
| Held-out trials | Training trials only (primary) | **69.69** |
| LOSO, latter half | Training subjects (strictly inductive) | 43.19 |
| LOSO, latter half | Past target samples (causal-online) | **73.59** |
| LOSO, latter half | Whole target session (transductive) | 74.30 |

The classifier is one fixed 1,720-parameter early-fusion logistic regression. It is not claimed as a new model;
the contribution is the evaluation protocol and the causal-online normalization.

## Repository layout

```
scripts/    experiment and figure scripts
results/    result files produced by those scripts (no SEED-V data)
figures/    Figures 1-4 of the paper (PNG and PDF)
data/       empty; put the SEED-V feature files here (see data/README.md)
```

## Data

SEED-V is distributed by the BCMI Laboratory, Shanghai Jiao Tong University, under an academic license and is
**not** included here. Request it from https://bcmi.sjtu.edu.cn/home/seed/seed-v.html and place the released
feature folders as described in `data/README.md`. Set the `SEEDV_DATA` environment variable to use another
location.

## Setup

```
pip install -r requirements.txt
```

## Reproducing the paper

Run the scripts from the repository root. Each writes into `results/`.

| Paper item | Script | Output |
|---|---|---|
| Table 2, Sec. III-B: modality ablation, paired tests, timing | `scripts/spmb_modality_ablation.py` | `results/cloud_instance/spmb_results/` |
| Table 1 row 1: segment-random split, and the training-size control (Sec. III-A) | `scripts/dcca_reconcile.py`, `scripts/methodology.py` | `results/workstation/RESULTS_dcca_reconciled.md`, `RESULTS_methodology.md` |
| Table 1 rows 2-3: held-out trials with whole-session vs training-only normalization | `scripts/trialindep_inductive.py` | `results/workstation/RESULTS_trialindep_inductive.md` |
| Table 3: 12-model benchmark (fixed, untuned settings) | `scripts/bench_inductive.py` | `results/workstation/RESULTS_benchmark_inductive.md` |
| Sec. III-C: DCCA reproduction | `scripts/dcca_inductive.py` | `results/workstation/RESULTS_dcca_inductive.md` |
| Sec. III-D: nested tuning and late fusion | `scripts/spmb_inductive_boost.py` | `results/cloud_instance/spmb_boost_results/` |
| Sec. III-D: classifiers on per-trial summary features | `scripts/spmb_direct_trial_models.py` | `results/cloud_instance/spmb_trial_results/` |
| Table 1 rows 4-6, Fig. 2, Sec. III-E: LOSO by normalization access | `scripts/loso_final.py` (also `cross_subject_audit.py`, `online_norm.py`) | `results/workstation/RESULTS_loso_*.md`, `loso_final_arrays.npz` |
| Sec. III-F: channel screening, five fold seeds, permutation importance | `scripts/inductive_fast.py`, `scripts/perm_compare.py` | `results/workstation/RESULTS_inductive_fast.md`, `RESULTS_perm_compare.md` |
| Fig. 3B, Fig. 4: electrode-selection stability, per-class metrics, class balance | `scripts/revision_analyses.py` | `results/revision/revision_analyses/` (all 48 fold-specific 10-channel selections: `sel10_fold_selections.csv`; selection frequency of all 62 electrodes: `sel10_selection_frequency.csv`) |
| Figures 1-4 | `scripts/make_fig1.py`, `make_fig2.py`, `make_fig3_fig4.py` | `figures/` |

The figure scripts need only the files in `results/`, so they run without SEED-V.

The causal-online normalization is the function `online_norm` in `scripts/loso_final.py`: running mean and
variance start from the pooled training-subject moments with pseudo-count 20, each segment is normalized and
classified with past segments only, and the running sums are updated afterwards.

## Hardware actually used

No experiment in the paper used GPU acceleration. No script selects a CUDA device, and the neural comparators
were trained with a CPU-only PyTorch build.

| Experiments | Machine | Evidence |
|---|---|---|
| Modality ablation (primary result), nested tuning / late fusion, per-trial models | CPU of a Linux cloud instance provided through the NVIDIA Academic Hardware Grant via Brev.dev (scikit-learn only) | `results/cloud_instance/spmb_results/spmb_efficiency.json` |
| 12-model benchmark, DCCA, LOSO analyses, channel screening, permutation importance | AMD workstation CPU, Windows 11, PyTorch 2.11.0+cpu, 8 threads | `results/workstation/ENVIRONMENT_FREEZE.md` |
| Timings reported in the paper, electrode-selection stability, per-class metrics, rerun of the modality ablation | Intel Core Ultra 9 285K (24 cores, 128 GB RAM), Windows 11, 8 BLAS threads | `results/revision/modality/spmb_efficiency.json` |

The rerun of the modality ablation on the Intel machine reproduces the primary 69.69% exactly
(`results/revision/modality/`).

## Reproduction check

All scripts were re-run from a fresh clone of this repository in a clean environment built from
`requirements.txt` (Python 3.12.10, Windows 11, Intel Core Ultra 9 285K, CPU only) and compared with the
published result files.

| Result | Published | Re-run |
|---|---|---|
| Modality ablation (Table 2), all metrics and paired tests | 69.69 / 65.30 / 51.96 | identical |
| Nested tuning, late fusion, per-trial models (Sec. III-D) | 69.66 / 65.87 / 63.75 / 67.64 | identical |
| Segment-random and whole-session normalization (Table 1) | 100.00 / 75.49 | identical |
| Benchmark (Table 3), 11 of 12 models including all neural comparators | as in Table 3 | identical |
| Benchmark, LDA | 36.29 | 36.35 |
| DCCA reproduction | 63.86 | 63.71 |
| LOSO causal-online and transductive | 73.59 / 74.30 | identical |
| LOSO strictly inductive baseline | 43.19 | 43.03 |
| Channel screening, five fold seeds, permutation importance, electrode stability, per-class metrics | as in the paper | identical |

The three small differences come from floating-point behaviour that depends on the CPU and BLAS library: the
LDA SVD solver, DCCA training, and the inductive LOSO logistic regression. None changes a conclusion. The paper
reports the original values; the re-run files are in `results/reproduction_check/`. Timings differ between
machines by design.

Convergence of the LOSO classifiers was checked separately (`scripts/loso_convergence_check.py`,
`results/reproduction_check/RESULTS_loso_convergence.md`). No fold reaches the 500-iteration limit: the strictly
inductive classifier converges in about 315 iterations and the session-aligned classifier in about 67, and
raising the limit to 20,000 changes no participant's accuracy. The 43.19 versus 43.03 difference is therefore a
cross-machine numerical difference, not an optimization artefact.

Correction (5 October 2026): `loso_final.py` originally applied the Holm multipliers without the step-down
running maximum, so one of two tied comparisons was reported as 9.16e-05. The corrected Holm-adjusted Wilcoxon
p-value for causal-online versus inductive is 1.22e-04, as stated in the paper. The script and the saved
summaries have been corrected; raw p-values and all accuracies are unchanged.

## Notes

- Comparator models use fixed settings chosen before evaluation and were not tuned per model; the benchmark
  ranking should be read with that in mind.
- The DCCA baseline is a simplified reimplementation, not a faithful reproduction of the published model: it
  differs in network size, output dimension, regularization and downstream classifier.
- Table 2 metrics are computed from pooled out-of-fold predictions; the benchmark file reports fold-averaged
  accuracy, balanced accuracy and F1, which should not be interchanged with the pooled values.
- Channels are ranked by the ANOVA F score averaged over the five bands of each electrode.
- The reported prediction time is batched classifier time divided by the number of segments. It is not a
  single-segment streaming latency and excludes feature extraction and normalization.
- "Causal" refers only to temporal information access (predict, then update); no causal inference is implied.
- Part [B] of `RESULTS_methodology.md` screens channels with whole-session normalization (an earlier
  diagnostic). The channel numbers in the paper are the training-only ones in `RESULTS_inductive_fast.md`.
- `RESULTS_dcca_reconciled.md` also uses whole-session normalization (the 75.49% setting); the leakage-safe DCCA
  comparison in the paper is `RESULTS_dcca_inductive.md`.
- Per-trial prediction dumps and console logs are not included.
- Software: Python 3.12-3.13, NumPy 2.4.2, SciPy 1.17.1, scikit-learn 1.8.0, PyTorch 2.11.0 (CPU), Matplotlib 3.10.

## Citation

```
Priyadharshini D and Shridevi S, "Leakage-Aware Causal Normalization for Lightweight Multimodal Emotion
Recognition on SEED-V," in Proc. IEEE Signal Processing in Medicine and Biology Symposium (SPMB), 2026.
```

## Acknowledgments

Vellore Institute of Technology, Chennai, for research support; the NVIDIA Academic Hardware Grant (via
Brev.dev, awarded to Shridevi S) for cloud compute; and the BCMI Laboratory, Shanghai Jiao Tong University, for
academic access to SEED-V.
