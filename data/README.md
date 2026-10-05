# SEED-V features (not included)

SEED-V is licensed by the BCMI Laboratory, Shanghai Jiao Tong University, and must be requested from
https://bcmi.sjtu.edu.cn/home/seed/seed-v.html.

After obtaining it, place the released feature folders here:

```
data/
  EEG_DE_features/        1_123.npz ... 16_123.npz   (310 differential-entropy features per 4-s segment)
  Eye_movement_features/  1_123.npz ... 16_123.npz   (33 eye-movement features per segment)
```

Each file holds the three sessions of one participant (45 trials). To keep the data elsewhere, set the
`SEEDV_DATA` environment variable to that folder, or pass `--data-root` to the `spmb_*.py` scripts.
