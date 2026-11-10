# Phase 5 tables

## A. Controlled comparison of few-shot heads

Identical v0+BN encoder, data, episodic budget (≤8,000 episodes, early stopping), optimiser and cross-view protocol; only the head differs. CASIA-B LT test, 5 seeds, mean ± 95% CI. Open-set: margin score for all heads (our model: likelihood ratio).

| Head | Params | FG-CV mean | FG-CV CL | S3 corrupted enroll | S4 occluded probes | OS AUROC NM | OS AUROC CL | OS DIR@1% NM | OS OSCR occluded |
|---|---|---|---|---|---|---|---|---|---|
| Prototypical (= L2) | 126k | 59.36 ± 1.82 | 39.66 | 33.29 ± 1.49 | 26.27 ± 0.98 | 73.50 ± 1.41 | 55.55 ± 0.92 | 19.07 ± 1.30 | 31.83 ± 1.00 |
| Matching (no FCE) | 126k | 49.90 ± 1.50 | 33.47 | 46.65 ± 1.64 | 19.94 ± 3.81 | 66.89 ± 1.10 | 54.77 ± 0.47 | 9.73 ± 2.30 | 22.77 ± 4.93 |
| Relation | 0.67M | 51.21 ± 3.89 | 34.96 | 31.21 ± 3.72 | 10.83 ± 1.63 | 67.91 ± 2.99 | 55.50 ± 1.77 | 13.53 ± 3.93 | 12.03 ± 1.81 |
| Meta-Baseline | 126k | 50.46 ± 1.60 | 29.44 | 28.03 ± 4.76 | 11.67 ± 3.74 | 67.44 ± 0.66 | 52.27 ± 0.80 | 11.43 ± 1.09 | 14.79 ± 5.23 |
| FEAT | 3.3M | 57.55 ± 4.61 | 39.96 | 45.06 ± 5.53 | 25.45 ± 3.64 | 72.02 ± 2.64 | 57.36 ± 0.61 | 17.52 ± 1.09 | 30.70 ± 3.81 |
| TADAM-style | 0.91M | 57.26 ± 1.16 | 39.28 | 29.52 ± 4.02 | 14.33 ± 2.50 | 72.04 ± 0.45 | 55.85 ± 0.31 | 14.49 ± 1.20 | 14.48 ± 2.75 |
| B1: Proto + per-strip aux (ours) | 126k | 66.07 ± 0.27 | 45.52 | 34.95 ± 3.74 | 28.92 ± 1.98 | 79.02 ± 0.94 | 58.04 ± 0.81 | 27.92 ± 1.66 | 34.72 ± 2.12 |
| Model of record: B1 + N1/N2/N4 occ-aware (ours) | 143k | 67.97 ± 0.65 | 49.93 | 66.55 ± 0.45 | 53.35 ± 0.81 | 82.24 ± 1.20 | 62.75 ± 2.15 | 34.67 ± 2.03 | 59.84 ± 0.81 |

## B. Frozen supervised backbones (OpenGait, CASIA-B LT) and the plug-in reliability head

Silhouette sequences; our few-shot protocols (prototype enrollment, no retraining). "+ head" = N1/N2 occlusion-aware variance head fitted on the 74 training subjects (5 head seeds). Open-set: raw max-score for every row.

| Model | Params | FG-CV mean | FG-CV CL | S1 mismatched | S3 corrupted | S4 occluded | OS AUROC NM | OS AUROC CL | OS DIR@1% NM | OS DIR@1% BG | OS AUROC occl. | OS DIR@1% occl. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaitbase frozen | 7.3M | 89.83 | 77.58 | 80.01 | 89.48 | 87.02 | 96.94 | 82.90 | 77.02 | 66.49 | 95.82 | 70.16 |
| gaitbase + head | 7.3M+16k | 89.63 ± 0.29 | 77.15 ± 0.74 | 79.96 ± 0.62 | 89.27 ± 0.28 | 86.59 ± 0.27 | 97.19 ± 0.06 | 82.46 ± 0.45 | 78.32 ± 0.29 | 68.40 ± 0.29 | 95.79 ± 0.07 | 70.62 ± 0.42 |
| gaitset frozen | 2.6M | 87.82 | 76.86 | 76.85 | 75.63 | 46.10 | 96.57 | 81.03 | 74.13 | 54.00 | 72.41 | 15.02 |
| gaitset + head | 2.6M+16k | 87.91 ± 0.11 | 76.97 ± 0.26 | 77.27 ± 0.17 | 87.11 ± 0.11 | 68.16 ± 0.13 | 96.41 ± 0.07 | 83.69 ± 0.18 | 74.75 ± 0.22 | 57.91 ± 0.24 | 85.70 ± 0.06 | 33.95 ± 0.11 |

## C. Cross-fit check: head fitted on 25 UNSEEN test subjects, evaluated on the other 25 (2 folds x 3 seeds; 25-subject gallery, so absolute values are higher than in B)

| Backbone | metric | frozen | + head | Δ |
|---|---|---|---|---|
| gaitbase | FG-CV mean | 92.23 | 92.82 | +0.59 |
| gaitbase | FG-CV CL | 83.28 | 84.81 | +1.53 |
| gaitbase | S3 corrupted | 91.91 | 92.73 | +0.82 |
| gaitbase | S4 occluded | 89.92 | 90.58 | +0.66 |
| gaitbase | OS AUROC CL | 87.47 | 88.49 | +1.02 |
| gaitbase | OS DIR@1% NM | 81.64 | 82.73 | +1.09 |
| gaitbase | OS AUROC occluded | 97.04 | 97.22 | +0.18 |
| gaitset | FG-CV mean | 90.98 | 91.59 | +0.62 |
| gaitset | FG-CV CL | 83.11 | 84.51 | +1.41 |
| gaitset | S3 corrupted | 80.65 | 90.05 | +9.40 |
| gaitset | S4 occluded | 55.51 | 69.04 | +13.53 |
| gaitset | OS AUROC CL | 85.63 | 88.59 | +2.96 |
| gaitset | OS DIR@1% NM | 74.71 | 74.18 | -0.52 |
| gaitset | OS AUROC occluded | 77.02 | 86.77 | +9.75 |
