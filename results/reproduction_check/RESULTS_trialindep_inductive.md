=== TRIAL-INDEPENDENT: session-transductive (current) vs training-only inductive norm ===
Model            trans%  induct%   delta
LogReg           75.49    69.69   -5.80
SVM-RBF          68.27    59.48   -8.80
LDA              57.96    36.35  -21.61
KNN              56.69    54.16   -2.53
RandomForest     65.84    61.96   -3.88
GradBoost        57.93    52.61   -5.32
NaiveBayes       58.18    51.45   -6.73
MLP              69.99    64.84   -5.15

Inductive LogReg: acc 69.69 | balanced-acc 70.19 | macroF1 0.697
Per-class recall (inductive, pooled): Disgust 60.8 | Fear 82.9 | Sad 59.5 | Neutral 71.6 | Happy 76.1
