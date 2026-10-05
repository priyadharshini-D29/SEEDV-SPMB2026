=== FINAL LOSO by normalisation access (16 subjects, COMMON last-50% window) ===
Inductive                 : 43.19 +/-  9.7
Calibration chrono-20%    : 47.78 +/-  7.8
Calibration repr-20% (2ndary): 64.77 +/-  9.4
Online n0=20 (pre-reg)    : 73.59 +/-  9.4
Online n0=nested          : 72.93 +/-  9.3
Transductive              : 74.30 +/-  8.2
Online full-session (n0=20): 69.27 +/-  8.2

Nested n0 picks across 16 folds: [20, 50, 50, 50, 50, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]  (mode=20)

Paired comparisons (online n0=20, last-50% window; Holm on Wilcoxon):
  online vs inductive   : diff +30.40 pts | 95%CI [+24.49,+36.38] | t p=6.33e-08 | Wilcoxon p=3.05e-05 | Holm p=9.16e-05 | improved 16/16
  online vs chrono-cal20: diff +25.81 pts | 95%CI [+20.38,+31.93] | t p=4.33e-07 | Wilcoxon p=3.05e-05 | Holm p=1.22e-04 | improved 16/16
  online vs repr-cal20  : diff +8.82 pts | 95%CI [+5.96,+11.47] | t p=2.46e-05 | Wilcoxon p=2.14e-04 | Holm p=4.27e-04 | improved 14/16
  online vs transductive: diff -0.71 pts | 95%CI [-3.35,+2.09] | t p=6.26e-01 | Wilcoxon p=4.33e-01 | Holm p=4.33e-01 | improved 6/16

Learning curve (online n0=20, accuracy % by session decile):
  0-10%:74.9 | 10-20%:61.5 | 20-30%:57.1 | 30-40%:63.8 | 40-50%:67.4 | 50-60%:73.8 | 60-70%:71.1 | 70-80%:80.9 | 80-90%:75.0 | 90-100%:67.1

Per-subject (inductive / online n0=20 last-50% / transductive):
  S01:  64.8 /  80.2 /  81.4
  S02:  42.1 /  73.6 /  72.5
  S03:  43.3 /  78.1 /  83.1
  S04:  42.3 /  84.2 /  89.4
  S05:  25.3 /  65.0 /  68.0
  S06:  43.7 /  68.9 /  65.1
  S07:  47.6 /  72.2 /  74.8
  S08:  54.9 /  75.5 /  66.5
  S09:  46.9 /  89.9 /  78.9
  S10:  34.9 /  48.8 /  54.7
  S11:  40.5 /  67.1 /  77.2
  S12:  32.1 /  83.9 /  84.1
  S13:  55.2 /  64.5 /  71.9
  S14:  30.0 /  77.5 /  76.6
  S15:  47.4 /  73.7 /  75.0
  S16:  40.0 /  74.3 /  69.8
