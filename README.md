# Know Which Body Part to Trust: a plug-in per-part reliability head for gait recognition

Code, trained weights and result files for the paper

> T. Riahi, H. A. Mustafa, *Know Which Body Part to Trust: A Plug-in Per-Part Reliability Head for Robust Few-Shot Gait Enrollment and Rejection*.

The reliability head attaches to any **frozen** part-based gait model. For every body part of every sample, it predicts a variance that says how far that part can be trusted. A single Gaussian likelihood then provides:

- **M1 – reliability-aware matching.** Part weights depend on both the probe and the enrolled identity.
- **M2 – precision-weighted enrollment.** Enrollment walks are fused part by part, so the occluded strip of one walk is discounted while its intact strips are kept. Enrollment is closed-form and incremental.
- **M3 – rejection of unknown walkers.** The same likelihood gives an acceptance score for open-set operation.
- **Exact explanations.** Every decision margin decomposes into per-part contributions.

The head has about 16k parameters. It is calibrated in a second stage on a small set of identities the backbone has never seen.

## Repository layout

| Path | Content |
|---|---|
| `src/gaitlab/` | all code (Python package; run modules with `PYTHONPATH=src`) |
| `src/gaitlab/models/` | B1 GEI backbone (`protonet.py`), reliability head, M1 matching and M2 enrollment (`probnet.py`), six few-shot heads (`heads.py`), attention baseline (`dppnet.py`) |
| `src/gaitlab/data/` | dataset caches, episodic sampler, CASIA-E / OU-MVLP dense loaders |
| `configs/` | YAML configs of every training run |
| `tests/` | unit tests for the protocols and the probabilistic model |
| `scripts/` | resumable drivers for the frozen-backbone experiments |
| `external/opengait_local.patch` | two small compatibility patches for OpenGait |
| `results/` | per-run JSON results used for every table of the paper, plus the final trained weights (below) |

### Included weights (about 7 MB)

| Path | Model |
|---|---|
| `results/p3_bn_aux/seed*/best.pt` | B1 backbone, CASIA-B (5 seeds) |
| `results/md_casiae_b1/seed*/best.pt`, `results/md_oumvlp_b1/seed*/best.pt` | B1 backbone, CASIA-E / OU-MVLP (3 seeds) |
| `results/md_{casiab,casiae,oumvlp}_head/seed*/var_head.pt` | calibrated reliability heads for B1 |
| `results/p5_plugin/head_{gaitset,gaitbase}_seed*.pt` | reliability heads for frozen GaitSet / GaitBase |

## Installation

```bash
git clone https://github.com/Tasrifur/gait-reliability-head.git
cd gait-reliability-head
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt        # install the PyTorch build for your CUDA version first
```

Set `PYTHONPATH=src` before running any module (Windows PowerShell: `$env:PYTHONPATH='src'`). On Windows, also set `KMP_DUPLICATE_LIB_OK=TRUE` if PyTorch and NumPy load different OpenMP runtimes.

## Data

The datasets are not redistributed. Request them from their providers (CASIA-B and CASIA-E from CASIA, OU-MVLP from OU-ISIR), then place the following in the repository root:

| File | Expected content |
|---|---|
| `casia_b_silhouettes.zip` | `<subject>/<seq>/<view>/*.png`, 64×64 cropped silhouettes (e.g. `001/nm-01/000/000.png`) |
| `casia_e_gei.zip` | CASIA-E GEIs, 88×128 |
| `oumvlp_gei.zip` | OU-MVLP GEIs, 88×128 |

Build the caches (about 1–2 h in total):

```bash
python -m gaitlab.data.build_casiab --zip casia_b_silhouettes.zip --out data/cache/casiab_gei.npz
python -m gaitlab.data.build_gei_zip --dataset casiae --zip casia_e_gei.zip --outdir data/cache
python -m gaitlab.data.build_gei_zip --dataset oumvlp --zip oumvlp_gei.zip --outdir data/cache
python -m gaitlab.data.dense
python -m pytest tests -q
```

## Reproducing the results

**B1 backbone and reliability head (CASIA-B, CASIA-E, OU-MVLP).** The included weights let you skip training:

```bash
python -m gaitlab.train --config configs/p3_bn_aux.yaml                 # B1 on CASIA-B (md_casiae_b1 / md_oumvlp_b1 for the others)
python -m gaitlab.calibrate --dataset casiab --b1 results/p3_bn_aux --out results/md_casiab_head
python -m gaitlab.multi_eval --dataset casiab                           # -> results/md_eval_casiab.json
python -m gaitlab.robust_baselines --dataset casiab                     # training-free baselines
python -m gaitlab.paper_analyses                                        # per-view, severity, DIR-FAR, reliability, cost
```

**Frozen official OpenGait checkpoints (GaitSet, GaitPart, GaitGL, GaitBase).**

```bash
git clone https://github.com/ShiqiYu/OpenGait.git external/opengait_src
cd external/opengait_src && git apply ../opengait_local.patch && cd ../..
```

Download the official CASIA-B checkpoints into `external/opengait_ckpt/CASIA-B/`:
- GaitSet, GaitPart and GaitGL: `pretrained_casiab_model.zip` from the OpenGait v1.0 release.
- GaitBase: `Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt` from the OpenGait Hugging Face page (`opengait/OpenGait`).

The expected paths are, for example, `external/opengait_ckpt/CASIA-B/GaitSet/GaitSet/checkpoints/GaitSet-40000.pt`. Then run:

```bash
python -m gaitlab.frozen_hybrid embed gaitset    # embeddings of every test sequence under 7 corruptions (resumable)
python -m gaitlab.frozen_hybrid eval gaitset     # head and head + energy prior, 2 folds x 5 seeds
python -m gaitlab.plugin_head final cuda gaitset # head on the frozen backbone (final protocol)
python -m gaitlab.frozen_wdist gaitset           # precision-weighted backbone distance
```

Replace `gaitset` with `gaitpart`, `gaitgl` or `gaitbase`. `scripts/run_frozen_hybrid.ps1` (Windows) and `scripts/run_frozen_extra.sh` (Linux) run these steps for several backbones and resume after an interruption. On a GPU a backbone takes minutes; on a CPU, several hours.

**Tables and figures.** Every number in the paper is generated from the JSON files in `results/`:

```bash
python -m gaitlab.latex_tables        # -> elsarticle/tables/*.tex
python -m gaitlab.latex_tables_extra  # -> elsarticle/figs/*.pdf
```

Only the dataset-statistics table and the two figures that show silhouettes (probe corruptions, per-part variance) need the dataset caches; they are skipped if the caches have not been built. Figures are vector PDFs with embedded fonts, sized for a two-column layout. Without the caches, `pytest` likewise skips the sampler tests.

## Citation

If you use this code, please cite the paper (citation details will be added on publication). The software itself is archived on Zenodo; citation metadata for it are in `CITATION.cff`, which GitHub shows under "Cite this repository".

## License

MIT; see `LICENSE`. The datasets and the OpenGait checkpoints are subject to their own licenses.
