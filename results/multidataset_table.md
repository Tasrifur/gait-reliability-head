# Multi-dataset results: B1 vs B1 + reliability head (calibrated on held-out ids)

Official test protocols, full-gallery cross-view. Head calibrated with horizontal bands only (vband / bottom / noise are unseen corruption types). Open-set rule chosen on the calibration ids. Mean ± 95% CI over training seeds; wins = seeds where the head beats B1.

## CASIA-B (5 seeds; open-set rules B1 ['raw', 'margin', 'margin', 'margin', 'raw'], head ['lr', 'lr', 'lr', 'lr', 'lr'])

| metric | B1 | + head | Δ | wins |
|---|---|---|---|---|
| FG-CV mean | 66.07 ± 0.27 | 67.86 ± 0.73 | +1.79 | 5/5 |
| FG-CV CL | 45.52 ± 1.08 | 50.27 ± 1.21 | +4.75 | 5/5 |
| Mismatched gallery | 36.75 ± 0.76 | 38.58 ± 1.40 | +1.83 | 5/5 |
| Corrupted enrollment | 34.95 ± 3.74 | 66.31 ± 0.82 | +31.37 | 5/5 |
| Occluded probes: hband | 28.92 ± 1.98 | 52.92 ± 1.01 | +24.00 | 5/5 |
| Occluded probes: vband | 5.14 ± 0.97 | 5.21 ± 0.92 | +0.07 | 3/5 |
| Occluded probes: bottom | 17.40 ± 1.39 | 33.83 ± 5.67 | +16.44 | 5/5 |
| Occluded probes: noise | 5.60 ± 0.86 | 3.77 ± 2.06 | -1.82 | 1/5 |
| Open-set bg AUROC | 70.10 ± 3.38 | 72.99 ± 2.51 | +2.89 | 3/5 |
| Open-set bg DIR@1% | 17.02 ± 2.33 | 19.53 ± 1.79 | +2.50 | 5/5 |
| Open-set cl AUROC | 58.15 ± 0.32 | 62.79 ± 1.97 | +4.64 | 5/5 |
| Open-set cl DIR@1% | 5.95 ± 1.18 | 8.30 ± 0.58 | +2.35 | 5/5 |
| Open-set nm AUROC | 78.86 ± 1.32 | 81.85 ± 1.18 | +3.00 | 5/5 |
| Open-set nm DIR@1% | 31.46 ± 4.67 | 34.14 ± 2.01 | +2.68 | 3/5 |
| Open-set occ_hband AUROC | 55.15 ± 5.55 | 71.97 ± 0.93 | +16.82 | 5/5 |
| Open-set occ_hband DIR@1% | 3.85 ± 3.11 | 15.63 ± 1.09 | +11.78 | 5/5 |

## CASIA-E (3 seeds; open-set rules B1 ['raw', 'margin', 'raw'], head ['lr', 'lr', 'lr'])

| metric | B1 | + head | Δ | wins |
|---|---|---|---|---|
| FG-CV mean | 24.35 ± 0.94 | 23.77 ± 1.46 | -0.58 | 0/3 |
| FG-CV CL | 16.15 ± 0.75 | 15.71 ± 1.22 | -0.44 | 0/3 |
| Mismatched gallery | 15.61 ± 0.59 | 15.26 ± 1.11 | -0.34 | 0/3 |
| Corrupted enrollment | 5.75 ± 0.65 | 20.27 ± 2.41 | +14.52 | 3/3 |
| Occluded probes: hband | 3.20 ± 1.38 | 12.48 ± 1.55 | +9.27 | 3/3 |
| Occluded probes: vband | 0.37 ± 0.04 | 0.37 ± 0.03 | -0.00 | 2/3 |
| Occluded probes: bottom | 2.92 ± 0.49 | 4.50 ± 1.36 | +1.58 | 3/3 |
| Occluded probes: noise | 0.35 ± 0.09 | 0.44 ± 0.20 | +0.09 | 3/3 |
| Open-set bg AUROC | 55.27 ± 0.95 | 55.25 ± 0.88 | -0.02 | 1/3 |
| Open-set bg DIR@1% | 2.09 ± 3.29 | 1.96 ± 0.45 | -0.13 | 2/3 |
| Open-set cl AUROC | 54.02 ± 3.18 | 53.88 ± 0.70 | -0.13 | 1/3 |
| Open-set cl DIR@1% | 1.20 ± 2.09 | 1.21 ± 0.32 | +0.01 | 2/3 |
| Open-set nm AUROC | 58.64 ± 0.77 | 58.22 ± 1.27 | -0.43 | 1/3 |
| Open-set nm DIR@1% | 4.17 ± 5.03 | 3.43 ± 0.90 | -0.74 | 2/3 |
| Open-set occ_hband AUROC | 50.29 ± 0.48 | 53.28 ± 0.82 | +2.99 | 3/3 |
| Open-set occ_hband DIR@1% | 0.23 ± 0.10 | 1.00 ± 0.25 | +0.77 | 3/3 |

## OU-MVLP (3 seeds; open-set rules B1 ['margin', 'margin', 'margin'], head ['lr', 'lr', 'lr'])

| metric | B1 | + head | Δ | wins |
|---|---|---|---|---|
| FG-CV mean | 40.03 ± 0.39 | 38.58 ± 1.24 | -1.44 | 0/3 |
| Corrupted enrollment | 23.11 ± 0.17 | 22.45 ± 0.70 | -0.66 | 0/3 |
| Occluded probes: hband | 2.46 ± 0.23 | 16.97 ± 1.20 | +14.51 | 3/3 |
| Occluded probes: vband | 0.69 ± 0.05 | 0.63 ± 0.09 | -0.06 | 0/3 |
| Occluded probes: bottom | 2.87 ± 1.23 | 11.68 ± 1.97 | +8.81 | 3/3 |
| Occluded probes: noise | 0.06 ± 0.01 | 0.08 ± 0.02 | +0.03 | 3/3 |
| Open-set nm AUROC | 61.92 ± 0.20 | 60.72 ± 0.62 | -1.20 | 0/3 |
| Open-set nm DIR@1% | 11.26 ± 0.09 | 6.34 ± 0.57 | -4.92 | 0/3 |
| Open-set occ_hband AUROC | 50.34 ± 0.19 | 53.87 ± 0.05 | +3.53 | 3/3 |
| Open-set occ_hband DIR@1% | 0.08 ± 0.11 | 1.38 ± 0.07 | +1.30 | 3/3 |

