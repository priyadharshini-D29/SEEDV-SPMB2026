# Inductive benchmark (trial-independent, training-only normalisation, 16 subjects)

Model                 Acc%  BalAcc%  MacroF1
LogReg              69.69    70.11    0.682
MLP                 66.53    66.73    0.641
RandomForest        61.96    61.84    0.597
Transformer         61.54    61.10    0.582
SVM-RBF             59.48    59.94    0.563
BiLSTM              57.91    57.01    0.533
1D-CNN              56.61    56.26    0.524
KNN                 54.16    53.52    0.513
GradBoost           52.61    52.09    0.496
NaiveBayes          51.45    51.81    0.472
CrossModal(ours)    49.26    48.99    0.464
LDA                 36.35    35.71    0.326

LogReg per-class recall (inductive): Disgust 60.8 | Fear 82.9 | Sad 59.5 | Neutral 71.6 | Happy 76.1
