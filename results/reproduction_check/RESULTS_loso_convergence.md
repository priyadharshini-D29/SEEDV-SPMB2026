# LOSO logistic-regression convergence check (latter-half accuracy, 16 held-out participants)

| Classifier | max_iter | folds stopped at the limit | mean iterations | accuracy (%) |
|---|---:|---:|---:|---:|
| Strictly inductive | 500 | 0/16 | 315 | 43.03 |
| Strictly inductive | 20000 | 0/16 | 315 | 43.03 |
| Session-aligned (transductive reference) | 500 | 0/16 | 67 | 74.30 |
| Session-aligned (transductive reference) | 20000 | 0/16 | 67 | 74.30 |

Largest per-participant change, inductive: 0.00 points; session-aligned: 0.00 points.
