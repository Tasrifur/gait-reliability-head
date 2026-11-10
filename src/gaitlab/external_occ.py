"""Frozen GaitBase/GaitSet + reliability head under the four corruption types
(hband, vband, bottom, noise), FINAL plug-in protocol (cross-fit calibration on the
unseen LT test subjects; see plugin_head.final_protocol).

Sequence corruption = the SAME geometry as the GEI evaluation: for each cell, the
cell's GEI is corrupted with occlusions.corrupt (same function, same salt), and the
difference is applied to every frame: pixels zeroed in the corrupted GEI are zeroed
in all frames; pixels raised (segmentation-noise blobs) are raised in all frames.

Stages:  embed  -> results/p5_frozen/embeddings_occ.pt
         eval   -> results/p5_plugin/final_occ_types.json
Usage: python -m gaitlab.external_occ embed|eval
"""
import io
import sys
import zipfile
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from .data.casiab import CasiaB, LT_GALLERY, LT_PROBES, SEQS, VIEWS
from .external_eval import build, dist
from .occlusions import TYPES, corrupt
from .plugin_head import BACKBONES, fit_head, fuse, nll_logits
from .utils import mean_ci, save_json


@torch.no_grad()
def embed(device='cuda'):
    ds = CasiaB('data/cache/casiab_gei.npz')
    cells, _ = torch.load('results/p5_frozen/embeddings.pt', weights_only=False)   # same cell order
    models = {n: build(n).to(device) for n in BACKBONES}
    zf = zipfile.ZipFile('casia_b_silhouettes.zip')
    groups = defaultdict(list)
    for n in zf.namelist():
        if n.endswith('.png'):
            s, q, v, _ = n.split('/')
            groups[(s, q, v)].append(n)
    out = {n: {t: [] for t in TYPES} for n in models}
    for i, (s, q, v) in enumerate(cells):
        files = sorted(groups[(ds.subjects[s], SEQS[q], VIEWS[v])])
        fr = np.stack([np.asarray(Image.open(io.BytesIO(zf.read(f))).convert('L'), np.float32) for f in files])
        fr = fr[(fr > 127).sum((1, 2)) >= 50][:720]
        gei = ds.get(s, q, v)                                       # [1,64,64] in [0,1]
        for t in TYPES:
            c = corrupt(gei, (s, q, v), t)[0]
            zero = (c == 0) & (gei[0] > 0)
            raise_ = c > gei[0] + 1e-6
            f2 = fr.copy()
            f2[:, zero] = 0
            f2[:, raise_] = np.maximum(f2[:, raise_], c[raise_] * 255.0)
            ten = torch.from_numpy(f2[:, :, 10:-10] / 255.0)[None, None].to(device)
            for n, m in models.items():
                out[n][t].append(m(ten)[0].cpu())
        if i % 500 == 0:
            print(f'{i}/{len(cells)} sequences', flush=True)
    torch.save({n: {t: torch.stack(v) for t, v in d.items()} for n, d in out.items()},
               'results/p5_frozen/embeddings_occ.pt')


@torch.no_grad()
def rank1(cells, M, V, subjects, pkey, kind, device):
    """Full-gallery cross-view rank-1 (gallery NM-01..04 clean, probes LT conditions,
    probe variant pkey) for one condition set; returns mean over NM/BG/CL."""
    idx = {c: i for i, c in enumerate(cells)}
    gq = [SEQS.index(x) for x in LT_GALLERY]
    res = []
    for cond in ['nm', 'bg', 'cl']:
        pq = [SEQS.index(x) for x in LT_PROBES[cond]]
        accs = []
        for pv in range(11):
            per = []
            for gv in range(11):
                if gv == pv:
                    continue
                ids = [s for s in subjects if any((s, x, gv) in idx for x in gq)]
                rows = [idx[(s, x, gv)] for s in ids for x in gq if (s, x, gv) in idx]
                owner = torch.tensor([ids.index(cells[r][0]) for r in rows], device=device)
                p = [idx[(s, x, pv)] for s in ids for x in pq if (s, x, pv) in idx]
                gm, qm = M['clean'][rows].to(device), M[pkey][p].to(device)
                if kind == 'frozen':
                    proto = torch.zeros(len(ids), *gm.shape[1:], device=device).index_add_(0, owner, gm)
                    proto /= torch.bincount(owner, minlength=len(ids))[:, None, None].float()
                    lg = -dist(qm, proto)
                else:
                    mc, vc = fuse(gm, V['clean'][rows].to(device), owner, len(ids))
                    lg = nll_logits(qm, V[pkey][p].to(device), mc, vc)
                pred = lg.argmax(1).cpu().numpy()
                per.append((np.array([ids[k] for k in pred]) == np.array([cells[i][0] for i in p])).mean() * 100)
            accs.append(np.mean(per))
        res.append(np.mean(accs))
    return float(np.mean(res))


def evaluate(device='cuda', seeds=(0, 1, 2, 3, 4)):
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    folds = [test[0::2], test[1::2]]
    cells, E = torch.load('results/p5_frozen/embeddings.pt', weights_only=False)
    EO = torch.load('results/p5_frozen/embeddings_occ.pt', weights_only=False)
    out = {}
    for bb in BACKBONES:
        Eb = {k: v.to(device) for k, v in E[bb].items()}
        M = {**{k: v for k, v in E[bb].items()}, **EO[bb]}
        out[bb] = {}
        for f in range(2):
            cal, ev = folds[f], folds[1 - f]
            r = {'frozen': {t: rank1(cells, M, None, ev, t, 'frozen', device) for t in ['clean'] + TYPES}, 'head': {}}
            for seed in seeds:
                h = fit_head(cells, Eb, cal, seed, ds, device, ['occ1', 'occ2'])   # identical to final_protocol
                with torch.no_grad():
                    V = {k: h(v.to(device)).cpu() for k, v in M.items()}
                r['head'][seed] = {t: rank1(cells, M, V, ev, t, 'head', device) for t in ['clean'] + TYPES}
                print(bb, 'fold', f, 'seed', seed, {t: round(x, 2) for t, x in r['head'][seed].items()},
                      '| frozen', {t: round(x, 2) for t, x in r['frozen'].items()}, flush=True)
            out[bb][f'fold{f}'] = r
            save_json(out, 'results/p5_plugin/final_occ_types.json')


if __name__ == '__main__':
    {'embed': embed, 'eval': evaluate}[sys.argv[1]]()
