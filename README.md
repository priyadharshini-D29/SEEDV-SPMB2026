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
| Sec. III-C: DCCA reproduction | `scripts/dcca_inductive.py`, `scripts/dcca_multiseed.py` | `results/workstation/RESULTS_dcca_*.md` |
| Sec. III-D: nested tuning and late fusion | `scripts/spmb_inductive_boost.py` | `results/cloud_instance/spmb_boost_results/` |
| Sec. III-D: classifiers on per-trial summary features | `scripts/spmb_direct_trial_models.py` | `results/cloud_instance/spmb_trial_results/` |
| Table 1 rows 4-6, Fig. 2, Sec. III-E: LOSO by normalization access | `scripts/loso_final.py` (also `cross_subject_audit.py`, `online_norm.py`) | `results/workstation/RESULTS_loso_*.md`, `loso_final_arrays.npz` |
| Sec. III-F: channel screening, five fold seeds, permutation importance | `scripts/inductive_fast.py`, `scripts/perm_compare.py` | `results/workstation/RESULTS_inductive_fast.md`, `RESULTS_perm_compare.md` |
| Fig. 3B, Fig. 4: electrode-selection stability, per-class metrics, class balance | `scripts/revision_analyses.py` | `results/revision/revision_analyses/` |
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

## Notes

- Comparator models use fixed settings chosen before evaluation and were not tuned per model; the benchmark
  ranking should be read with that in mind.
- "Causal" refers only to temporal information access (predict, then update); no causal inference is implied.
- Part [B] of `RESULTS_methodology.md` screens channels with whole-session normalization (an earlier
  diagnostic). The channel numbers in the paper are the training-only ones in `RESULTS_inductive_fast.md`.
- `RESULTS_dcca_reconciled.md` and `RESULTS_dcca_multiseed.md` also use whole-session normalization (the 75.49%
  setting); the leakage-safe DCCA comparison in the paper is `RESULTS_dcca_inductive.md`.
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
