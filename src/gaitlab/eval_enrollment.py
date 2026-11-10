"""Phase 4b: enrollment-scenario evaluation of B1 vs N1 vs N1+N2.

All scenarios: full-gallery cross-view rank-1 on the 50 LT test subjects
(identical view excluded); gallery and probe never share a sequence id.

  S1 condition matrix : gallery {NM-01..04 | BG-01 | CL-01} x probe {NM-05/06 | BG-02 | CL-02}
  S2 mixed enrollment : gallery {NM-01, NM-02, BG-01, CL-01}; probes NM-05/06, BG-02, CL-02
  S3 corrupted enroll : gallery NM-01..04, each support band-occluded with p=0.5; standard probes
  S4 occluded probes  : clean NM-01..04 gallery; every probe band-occluded; standard probes
Occlusions are deterministic per (cell, salt), so every model sees identical inputs.

Usage: python -m gaitlab.eval_enrollment --stage2 results/p4_2stage results/p4_2stage_occ
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from .data.casiab import CasiaB, LT_GALLERY, LT_PROBES
from .models.probnet import ProbProtoNet
from .train_uncertainty import make_variant
from .utils import mean_ci, save_json

GAL = {'nm': LT_GALLERY, 'bg': ['bg-01'], 'cl': ['cl-01']}
PRB = {'nm': ['nm-05', 'nm-06'], 'bg': ['bg-02'], 'cl': ['cl-02']}
MIXED = ['nm-01', 'nm-02', 'bg-01', 'cl-01']
VARIANT_NAMES = ['b1', 'n1_lam1', 'n1n2_lam1']


def band_for(cell, salt, p):
    """Deterministic occlusion decision for a cell: (top, height) or None."""
    rng = np.random.RandomState((hash(cell) + salt) % (2 ** 31))
    if rng.rand() >= p:
        return None
    h = rng.randint(6, 17)
    return rng.randint(0, 64 - h + 1), h


def occlude(img, cell, salt, p):
    """Deterministic band occlusion of one GEI [1,64,64] (numpy). The same band
    is applied to every frame of a silhouette sequence for sequence models."""
    b = band_for(cell, salt, p)
    if b is None:
        return img
    t, h = b
    img = img.copy()
    img[..., t:t + h, :] = 0
    return img


@torch.no_grad()
def fg_rank1(model, ds, subjects, gseqs, pseqs, device, g_occ=0.0, p_occ=0.0):
    gq, pq = [ds.qidx(q) for q in gseqs], [ds.qidx(q) for q in pseqs]
    accs = []
    for pv in range(11):
        per_gv = []
        for gv in range(11):
            if gv == pv:
                continue
            ids = [s for s in subjects if any(ds.valid[s, q, gv] for q in gq)]
            sup = [(s, q, gv) for s in ids for q in gq if ds.valid[s, q, gv]]
            qry = [(s, q, pv) for s in ids for q in pq if ds.valid[s, q, pv]]
            if not qry:
                continue
            X = torch.from_numpy(np.stack([occlude(ds.get(*c), c, 1, g_occ) for c in sup])).to(device)
            Q = torch.from_numpy(np.stack([occlude(ds.get(*c), c, 2, p_occ) for c in qry])).to(device)
            owner = torch.tensor([ids.index(c[0]) for c in sup], device=device)
            lab = np.array([ids.index(c[0]) for c in qry])
            logits = model.logits_unequal(X, owner, len(ids), Q)
            per_gv.append((logits.argmax(1).cpu().numpy() == lab).mean() * 100)
        accs.append(np.mean(per_gv))
    return float(np.mean(accs))


def scenarios(model, ds, test, device):
    r = {}
    for g in ['nm', 'bg', 'cl']:
        for p in ['nm', 'bg', 'cl']:
            r[f'S1_{g}>{p}'] = fg_rank1(model, ds, test, GAL[g], PRB[p], device)
    for p in ['nm', 'bg', 'cl']:
        r[f'S2_mixed>{p}'] = fg_rank1(model, ds, test, MIXED, PRB[p], device)
        r[f'S3_corrupt>{p}'] = fg_rank1(model, ds, test, LT_GALLERY, LT_PROBES[p], device, g_occ=0.5)
        r[f'S4_occprobe>{p}'] = fg_rank1(model, ds, test, LT_GALLERY, LT_PROBES[p], device, p_occ=1.0)
    return r


def load_pm(stage2_dir, seed, device):
    pm = ProbProtoNet('v0bn').to(device).eval()
    b1 = torch.load(f'results/p3_bn_aux/seed{seed}/best.pt', weights_only=True)
    pm.cnn.load_state_dict({k[4:]: v for k, v in b1.items() if k.startswith('cnn.')})
    pm.load_state_dict(torch.load(os.path.join(stage2_dir, f'seed{seed}', 'var_head.pt'),
                                  weights_only=True), strict=False)
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2', nargs='+', required=True)
    ap.add_argument('--out', default='results/p4b_enrollment')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    seeds = sorted(int(os.path.basename(d)[4:]) for d in glob.glob(os.path.join(a.stage2[0], 'seed*')))
    allres = {}
    for seed in seeds:
        res = {}
        for k, d in enumerate(a.stage2):
            pm = load_pm(d, seed, a.device)
            tag = os.path.basename(d.rstrip('/'))
            for v in VARIANT_NAMES:
                if v == 'b1' and k > 0:
                    continue                          # B1 does not depend on the variance head
                name = 'b1' if v == 'b1' else f'{tag}:{v}'
                print(f'seed {seed} {name}', flush=True)
                res[name] = scenarios(make_variant(pm, v), ds, test, a.device)
        allres[seed] = res
        save_json(allres, os.path.join(a.out, 'per_seed.json'))
    # aggregate
    models = list(allres[seeds[0]].keys())
    keys = list(allres[seeds[0]][models[0]].keys())
    summ = {m: {k: mean_ci([allres[s][m][k] for s in seeds]) for k in keys} for m in models}
    save_json(summ, os.path.join(a.out, 'summary.json'))
    lines = ['| scenario | ' + ' | '.join(models) + ' |', '|---' * (len(models) + 1) + '|']
    for k in keys:
        lines.append(f'| {k} | ' + ' | '.join(f"{summ[m][k]['mean']:.2f} ± {summ[m][k]['ci95']:.2f}"
                                              for m in models) + ' |')
    md = '\n'.join(lines)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
