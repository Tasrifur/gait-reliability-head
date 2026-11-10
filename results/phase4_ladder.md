# Phase 4 ablation ladder (CASIA-B LT, cross-view, 5 seeds, mean ± 95% CI)

| Row | Model | Params | FG-CV mean | FG-CV CL | S1 mismatched gallery | S3 corrupted enroll | S4 occluded probes | OS AUROC NM | OS AUROC CL | OS DIR@1% NM | OS OSCR occluded | Expl. separation CL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | Original DPP-Net (residual attention) | 685k | 53.76 ± 3.07 | 37.96 | — | 27.49 ± 1.65 | 15.89 ± 4.55 | 65.89 ± 1.84 | 53.69 ± 0.76 | 8.18 ± 0.95 | 14.23 ± 2.39 | 0.03 ± 0.02 |
| L1 | Uniform weights (from scratch) | 126k | 55.72 ± 1.27 | 37.23 | — | 29.11 ± 4.47 | 16.17 ± 1.68 | 71.26 ± 0.56 | 54.36 ± 1.23 | 13.83 ± 1.23 | 18.41 ± 1.96 | — |
| L2 | + BatchNorm | 126k | 59.36 ± 1.82 | 39.66 | — | 33.29 ± 1.49 | 26.27 ± 0.98 | 73.50 ± 1.41 | 55.55 ± 0.92 | 19.07 ± 1.30 | 31.83 ± 1.00 | — |
| L3 | + per-strip aux loss = **B1** | 126k | 66.07 ± 0.27 | 45.52 | 44.05 ± 0.62 | 34.95 ± 3.74 | 28.92 ± 1.98 | 79.02 ± 0.94 | 58.04 ± 0.81 | 27.92 ± 1.66 | 34.72 ± 2.12 | 0.06 ± 0.02 |
| L4 | + N1 per-strip uncertainty (two-stage) | 143k | 68.01 ± 0.36 | 49.92 | 46.97 ± 1.07 | 36.16 ± 3.04 | 18.79 ± 5.91 | — | — | — | — | — |
| L5 | + N2 precision-weighted prototypes | 143k | 67.98 ± 0.63 | 49.71 | 46.89 ± 1.30 | 32.86 ± 12.60 | 16.94 ± 7.18 | 82.03 ± 1.00 | 62.36 ± 2.34 | 36.19 ± 1.26 | 21.81 ± 6.91 | — |
| L6 | + occlusion-aware uncertainty (**model of record**) | 143k | 67.97 ± 0.65 | 49.93 | 46.60 ± 0.95 | 66.55 ± 0.45 | 53.35 ± 0.81 | 82.24 ± 1.20 | 62.75 ± 2.15 | 34.67 ± 2.03 | 59.84 ± 0.81 | 0.26 ± 0.03 |

FG-CV: full-gallery cross-view rank-1 (standard LT). S1/S3/S4: enrollment scenarios (mean over NM/BG/CL probes). OS: open-set, 25 enrolled / 25 unknown x 10 splits, best score rule per model (margin for L0-L3, likelihood ratio for L5-L6); L4 has no separate open-set run. Expl. separation: deletion flip-AUC (most-important-first minus least-important-first), CL probes; higher = more discriminative explanations.
