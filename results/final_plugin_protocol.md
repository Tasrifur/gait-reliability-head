# Final plug-in protocol: frozen official backbones + reliability head calibrated on unseen subjects

CASIA-B LT test subjects (075-124, never seen by the backbones) split into 2 folds. Per fold: head fitted and open-set score rule selected on the CALIBRATION fold only; all metrics on the EVALUATION fold (25-subject gallery). Mean over the 2 evaluation folds; head ± = 95% CI over 10 (fold, seed) runs; "wins" = runs in which the head beats the frozen model on the same evaluation fold.

## gaitbase

Selected open-set rule: frozen ['raw', 'raw']; head {'raw': 5, 'lr': 5}

| metric | frozen | + head | Δ | wins |
|---|---|---|---|---|
| FG-CV mean | 92.23 | 92.80 ± 0.72 | +0.56 | 10/10 |
| FG-CV CL | 83.28 | 84.73 ± 2.15 | +1.45 | 10/10 |
| S1 mismatched gallery | 84.97 | 86.11 ± 1.66 | +1.14 | 9/10 |
| S3 corrupted enrollment | 91.91 | 92.70 ± 0.54 | +0.79 | 10/10 |
| S4 occluded probes | 89.92 | 90.56 ± 0.50 | +0.64 | 10/10 |
| Open-set NM AUROC | 97.75 | 97.73 ± 0.27 | -0.02 | 5/10 |
| Open-set NM OSCR | 97.00 | 96.89 ± 0.40 | -0.12 | 5/10 |
| Open-set NM DIR@1% | 81.64 | 77.84 ± 1.15 | -3.80 | 5/10 |
| Open-set BG AUROC | 96.27 | 96.31 ± 0.05 | +0.04 | 5/10 |
| Open-set BG OSCR | 93.97 | 94.24 ± 0.28 | +0.28 | 5/10 |
| Open-set BG DIR@1% | 70.35 | 68.89 ± 3.06 | -1.46 | 5/10 |
| Open-set CL AUROC | 87.47 | 87.85 ± 1.56 | +0.38 | 5/10 |
| Open-set CL OSCR | 80.23 | 81.28 ± 1.89 | +1.05 | 9/10 |
| Open-set CL DIR@1% | 40.30 | 37.44 ± 1.19 | -2.85 | 0/10 |
| Open-set NM occluded AUROC | 97.04 | 96.80 ± 0.23 | -0.23 | 5/10 |
| Open-set NM occluded OSCR | 95.88 | 95.61 ± 0.46 | -0.28 | 5/10 |
| Open-set NM occluded DIR@1% | 76.80 | 71.34 ± 2.50 | -5.46 | 5/10 |

## gaitset

Selected open-set rule: frozen ['raw', 'raw']; head {'raw': 5, 'lr': 5}

| metric | frozen | + head | Δ | wins |
|---|---|---|---|---|
| FG-CV mean | 90.98 | 91.56 ± 0.63 | +0.59 | 10/10 |
| FG-CV CL | 83.11 | 84.43 ± 1.38 | +1.33 | 8/10 |
| S1 mismatched gallery | 82.63 | 83.67 ± 1.59 | +1.04 | 10/10 |
| S3 corrupted enrollment | 80.65 | 90.03 ± 0.20 | +9.38 | 10/10 |
| S4 occluded probes | 55.51 | 68.97 ± 1.14 | +13.45 | 10/10 |
| Open-set NM AUROC | 97.25 | 97.03 ± 0.08 | -0.23 | 5/10 |
| Open-set NM OSCR | 95.92 | 95.67 ± 0.12 | -0.25 | 5/10 |
| Open-set NM DIR@1% | 74.71 | 68.10 ± 4.45 | -6.61 | 5/10 |
| Open-set BG AUROC | 94.11 | 94.87 ± 0.05 | +0.76 | 10/10 |
| Open-set BG OSCR | 90.46 | 91.22 ± 0.19 | +0.75 | 10/10 |
| Open-set BG DIR@1% | 55.94 | 56.46 ± 4.57 | +0.52 | 7/10 |
| Open-set CL AUROC | 85.63 | 87.80 ± 0.19 | +2.17 | 10/10 |
| Open-set CL OSCR | 77.94 | 80.22 ± 0.57 | +2.28 | 10/10 |
| Open-set CL DIR@1% | 31.58 | 32.06 ± 2.19 | +0.49 | 6/10 |
| Open-set NM occluded AUROC | 77.02 | 85.82 ± 0.29 | +8.80 | 10/10 |
| Open-set NM occluded OSCR | 61.87 | 75.83 ± 0.15 | +13.96 | 10/10 |
| Open-set NM occluded DIR@1% | 22.59 | 29.44 ± 2.68 | +6.85 | 10/10 |

