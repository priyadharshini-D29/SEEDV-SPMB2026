# SPMB Direct Trial-Representation Study

## Evaluation safeguards

- Complete outer test trials are held out.
- EEG and eye normalization uses training trials only.
- Inner-fold normalization is refitted using inner-training trials.
- Scaling, feature selection and PCA occur inside training pipelines.
- Hyperparameters and the primary model family are selected without outer-test labels.

## Trial-level results

| Method | Accuracy (%) | Balanced acc. (%) | Macro-F1 | MCC |
|---|---:|---:|---:|---:|
| Segment-probability averaging reference | 67.64 | 67.64 | 0.675 | 0.596 |
| Trial mean + Logistic Regression | 67.22 | 67.22 | 0.670 | 0.591 |
| Trial mean/std + selected-feature LogReg | 65.56 | 65.56 | 0.654 | 0.570 |
| Trial moments + selected-feature LogReg | 62.08 | 62.08 | 0.619 | 0.526 |
| Trial mean + PCA-RBF-SVM | 58.06 | 58.06 | 0.580 | 0.476 |
| Trial mean + shrinkage LDA | 66.81 | 66.81 | 0.666 | 0.587 |
| Trial moments + Extra Trees | 56.53 | 56.53 | 0.563 | 0.457 |
| Nested-selected direct-trial model | 63.75 | 63.75 | 0.636 | 0.547 |

## Paired comparisons against the reference

| Direct-trial method | Mean Δ (pp) | 95% CI (pp) | Cohen dz | Raw p | Holm p | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| Trial mean + Logistic Regression | -0.42 | [-2.31, +1.48] | -0.117 | 0.4771 | 0.9542 | 3/4/9 |
| Trial mean/std + selected-feature LogReg | -2.08 | [-6.13, +1.96] | -0.275 | 0.2927 | 0.8781 | 6/1/9 |
| Trial moments + selected-feature LogReg | -5.56 | [-9.17, -1.94] | -0.818 | 0.008848 | 0.04107 | 3/0/13 |
| Trial mean + PCA-RBF-SVM | -9.58 | [-13.86, -5.31] | -1.195 | 0.001953 | 0.01172 | 2/1/13 |
| Trial mean + shrinkage LDA | -0.83 | [-4.57, +2.91] | -0.119 | 0.7764 | 0.9542 | 7/2/7 |
| Trial moments + Extra Trees | -11.11 | [-15.52, -6.70] | -1.343 | 0.0009801 | 0.006861 | 1/1/14 |
| Nested-selected direct-trial model | -3.89 | [-6.39, -1.39] | -0.828 | 0.008214 | 0.04107 | 1/2/13 |

## Nested selector choices

- Trial mean + Logistic Regression: 23 of 48 outer folds.
- Trial mean/std + selected-feature LogReg: 10 of 48 outer folds.
- Trial moments + selected-feature LogReg: 6 of 48 outer folds.
- Trial mean + PCA-RBF-SVM: 4 of 48 outer folds.
- Trial mean + shrinkage LDA: 4 of 48 outer folds.
- Trial moments + Extra Trees: 1 of 48 outer folds.

## Validation and decision

- Reproduced probability-averaging reference: 67.64%.
- Difference from the previous rounded 67.64%: -0.00 percentage points.
- Primary nested-selector accuracy: 63.75%.
- The reference reproduces within the predeclared ±0.30 percentage-point tolerance.
- The predeclared primary nested selector did not cross 80%. Do not select another family post hoc as the proposed method.

## Nested-selector per-class recall

- Disgust: 54.2%
- Fear: 81.9%
- Sad: 56.9%
- Neutral: 61.1%
- Happy: 64.6%

Software:
Python 3.12.13, NumPy 2.4.2, SciPy 1.17.1, scikit-learn 1.8.0.
