# SPMB Leakage-Safe Inductive Boost Study

## Evaluation safeguards

- Outer test trials are completely held out.
- Session normalization uses outer-training trials only.
- Hyperparameters are selected by grouped inner cross-validation.
- Inner-fold normalization is recomputed from inner-training trials only.
- Trial accuracy averages class probabilities within each held-out trial.
- Segment-level and trial-level results are reported separately.

## Results

| Endpoint | Method | Parameters | Accuracy (%) | Balanced acc. (%) | Macro-F1 | MCC |
|---|---|---:|---:|---:|---:|---:|
| Segment | Early fusion, C=1 | 1720 | 69.69 | 70.19 | 0.697 | 0.620 |
| Segment | Nested-tuned early fusion | 1720 | 69.66 | 69.96 | 0.696 | 0.619 |
| Segment | Nested-tuned late fusion | 1725 | 65.87 | 66.22 | 0.656 | 0.573 |
| Trial | Early fusion, C=1 | 1720 | 67.64 | 67.64 | 0.675 | 0.596 |
| Trial | Nested-tuned early fusion | 1720 | 66.67 | 66.67 | 0.666 | 0.584 |
| Trial | Nested-tuned late fusion | 1725 | 63.47 | 63.47 | 0.633 | 0.544 |

## Paired subject-level comparisons

| Endpoint | Comparison | Mean Δ (pp) | 95% CI (pp) | Cohen dz | Raw p | Holm p | W/T/L |
|---|---|---:|---:|---:|---:|---:|---:|
| Segment | Nested-tuned early fusion vs Early fusion, C=1 | -0.04 | [-1.13, +1.05] | -0.018 | 0.6832 | 0.6832 | 7/2/7 |
| Segment | Nested-tuned late fusion vs Early fusion, C=1 | -3.82 | [-6.89, -0.75] | -0.663 | 0.009186 | 0.04593 | 3/0/13 |
| Trial | Nested-tuned early fusion vs Early fusion, C=1 | -0.97 | [-2.65, +0.70] | -0.310 | 0.159 | 0.318 | 5/3/8 |
| Trial | Nested-tuned late fusion vs Early fusion, C=1 | -4.17 | [-7.16, -1.17] | -0.742 | 0.01187 | 0.04746 | 2/4/10 |
| Trial | Nested-tuned late fusion vs Nested-tuned early fusion | -3.19 | [-5.79, -0.60] | -0.656 | 0.01867 | 0.05601 | 2/6/8 |

## Validation and interpretation

- Baseline segment accuracy: 69.69%.
- Difference from the manuscript's rounded 69.7%: -0.01 percentage points.
- Best trial-level method: Early fusion, C=1.
- Best leakage-safe trial-level accuracy: 67.64%.
- Median selected early-fusion C: 1.
- Median selected late-fusion EEG weight: 0.60.
- The inductive baseline reproduces within the predeclared ±0.30 percentage-point tolerance.
- The 80% threshold was not crossed under this leakage-safe study.

## Best trial-level per-class recall

- Disgust: 58.3%
- Fear: 81.2%
- Sad: 61.1%
- Neutral: 66.7%
- Happy: 70.8%

Software:
Python 3.12.13, NumPy 2.4.2, SciPy 1.17.1, scikit-learn 1.8.0.
