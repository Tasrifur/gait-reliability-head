"""Silhouette-energy prior (the 'hybrid' of robust_baselines) and the energy-mask baseline on
frozen GaitSet / GaitBase, FINAL plug-in protocol (2-fold cross-fit over the unseen LT test
subjects, 5 head seeds per fold; identical head fitting to plugin_head.final_protocol).

Part energy: e_i = mean input intensity (cropped 64x44 frames, averaged over frames) over the rows
covered by part i (HPP bins mapped back to input rows; GaitBase 16 bins, GaitSet 2 x [16,8,4,2,1]).
r_i = clip(e_i / e_bar_i, 0, 1), e_bar_i = mean over clean CALIBRATION-fold cells.
  energy  : r-weighted prototypes, per-part Euclidean distance weighted by r_q,i * max_k r_k,i
  hybrid  : head variances s_i <- s_i / max(r_i, 1e-3) (gallery and probe), N1 + N2 as for 'head'
Rules (soft | r > 0.25/0.5/0.75) are selected per fold on the calibration fold
(clean, corrupted enrollment, horizontal band), separately for energy and hybrid.

Variants: clean, occ1/occ2 (bands of external_eval, used to fit the head), and the four
corruption types of external_occ (same frame-level geometry).

Stages:  embed -> results/p5_frozen/hyb_<bb>_<variant>.pt, hyb_energy_<variant>.pt (resumable)
         eval  -> results/p5_plugin/frozen_hybrid.json
Usage: python -m gaitlab.frozen_hybrid embed [gaitset|gaitbase] | eval [gaitset|gaitbase]
"""
import io
import os
import sys
import time
import zipfile
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from .data.casiab import CasiaB, LT_GALLERY, LT_PROBES, SEQS, VIEWS
from .eval_enrollment import band_for
from .external_eval import build, dist
from .occlusions import corrupt
from .plugin_head import fit_head, fuse, nll_logits
from .utils import save_json

VARIANTS = ['clean', 'occ1', 'occ2', 'hband', 'bottom', 'vband', 'noise']
RULES = ['soft', 0.25, 0.5, 0.75]
OUT = 'results/p5_frozen'
RES = 'results/p5_plugin/frozen_hybrid.json'


def part_rows(bb):
    """Input-row ranges [(lo, hi)] of each embedding part, in the model's part order."""
    levels = {'gaitbase': [16], 'gaitpart': [16], 'gaitgl': [64]}.get(bb, [16, 8, 4, 2, 1] * 2)
    return [(b * 64 // L, (b + 1) * 64 // L) for L in levels for b in range(L)]


def cells_and_ds():
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    cells = [(s, q, v) for s in test for q in range(10) for v in range(11) if ds.valid[s, q, v]]
    return ds, test, cells


def variant(fr, gei, cell, var):
    if var == 'clean':
        return fr
    f2 = fr.copy()
    if var in ('occ1', 'occ2'):
        b = band_for(cell, *{'occ1': (1, 0.5), 'occ2': (2, 1.0)}[var])
        if b is not None:
            f2[:, b[0]:b[0] + b[1], :] = 0
        return f2
    c = corrupt(gei, cell, var)[0]
    zero = (c == 0) & (gei[0] > 0)
    raise_ = c > gei[0] + 1e-6
    f2[:, zero] = 0
    f2[:, raise_] = np.maximum(f2[:, raise_], c[raise_] * 255.0)
    return f2


@torch.no_grad()
def embed(bb):
    torch.set_num_threads(14)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds, _, cells = cells_and_ds()
    m = build(bb).to(dev)
    zf = zipfile.ZipFile('casia_b_silhouettes.zip')
    groups = defaultdict(list)
    for n in zf.namelist():
        if n.endswith('.png'):
            s, q, v, _ = n.split('/')
            groups[(s, q, v)].append(n)
    for var in VARIANTS:
        path, epath = f'{OUT}/hyb_{bb}_{var}.pt', f'{OUT}/hyb_energy_{var}.pt'
        if os.path.exists(path):
            continue
        cdir, chunk = path + '.chunks', 250                  # resumable: one file per 250 cells
        os.makedirs(cdir, exist_ok=True)
        t0 = time.time()
        for k in range(0, len(cells), chunk):
            if os.path.exists(f'{cdir}/{k:05d}.pt'):
                continue
            emb, eng = [], []
            for s, q, v in cells[k:k + chunk]:
                files = sorted(groups[(ds.subjects[s], SEQS[q], VIEWS[v])])
                fr = np.stack([np.asarray(Image.open(io.BytesIO(zf.read(f))).convert('L'), np.float32)
                               for f in files])
                fr = fr[(fr > 127).sum((1, 2)) >= 50][:720]
                x = variant(fr, ds.get(s, q, v), (s, q, v), var)[:, :, 10:-10] / 255.0
                eng.append(torch.from_numpy(x.mean((0, 2)).astype(np.float32)))      # [64] row energy
                emb.append(m(torch.from_numpy(x)[None, None].float().to(dev))[0].cpu())
            torch.save((torch.stack(emb), torch.stack(eng)), f'{cdir}/{k:05d}.pt.tmp')
            os.replace(f'{cdir}/{k:05d}.pt.tmp', f'{cdir}/{k:05d}.pt')
            print(f'{bb}/{var}: {min(k + chunk, len(cells))}/{len(cells)} ({(time.time() - t0) / 60:.0f} min)',
                  flush=True)
        parts = [torch.load(f'{cdir}/{k:05d}.pt') for k in range(0, len(cells), chunk)]
        torch.save(torch.cat([p[0] for p in parts]), path)
        if not os.path.exists(epath):
            torch.save(torch.cat([p[1] for p in parts]), epath)
        for k in range(0, len(cells), chunk):
            os.remove(f'{cdir}/{k:05d}.pt')
        os.rmdir(cdir)
        print(f'{bb}/{var} done', flush=True)


def reliability(e, ebar, rule):
    r = (e / ebar).clamp(0, 1)
    return r if rule == 'soft' else (r > rule).float()


@torch.no_grad()
def rank1(cells, subjects, kind, M, V, R, gkey, pkey, conds=('nm', 'bg', 'cl')):
    """Full-gallery cross-view rank-1 as external_occ.rank1, with gallery variant gkey."""
    idx = {c: i for i, c in enumerate(cells)}
    gq = [SEQS.index(x) for x in LT_GALLERY]
    res = []
    for cond in conds:
        pq = [SEQS.index(x) for x in LT_PROBES[cond]]
        accs = []
        for pv in range(11):
            per = []
            for gv in range(11):
                if gv == pv:
                    continue
                ids = [s for s in subjects if any((s, x, gv) in idx for x in gq)]
                rows = [idx[(s, x, gv)] for s in ids for x in gq if (s, x, gv) in idx]
                owner = torch.tensor([ids.index(cells[r][0]) for r in rows])
                p = [idx[(s, x, pv)] for s in ids for x in pq if (s, x, pv) in idx]
                gm, qm = M[gkey][rows], M[pkey][p]
                if kind == 'frozen':
                    proto = torch.zeros(len(ids), *gm.shape[1:]).index_add_(0, owner, gm)
                    proto /= torch.bincount(owner, minlength=len(ids))[:, None, None].float()
                    lg = -dist(qm, proto)
                elif kind == 'energy':
                    gr, qr = R[gkey][rows], R[pkey][p]
                    w = torch.zeros(len(ids), gr.shape[1]).index_add_(0, owner, gr)
                    proto = torch.zeros(len(ids), *gm.shape[1:]).index_add_(0, owner, gm * gr[:, None])
                    proto /= w.clamp(min=1e-6)[:, None]
                    rc = torch.zeros(len(ids), gr.shape[1]).index_reduce_(0, owner, gr, 'amax', include_self=False)
                    d = torch.sqrt(((qm[:, None] - proto[None]) ** 2).sum(2).clamp(min=0))       # [q,n,P]
                    ww = qr[:, None] * rc[None]
                    lg = -(ww * d).sum(-1) / ww.sum(-1).clamp(min=1e-6)
                else:
                    gv_, qv_ = V[gkey][rows], V[pkey][p]
                    if kind == 'hybrid':
                        gv_, qv_ = gv_ / R[gkey][rows].clamp(min=1e-3), qv_ / R[pkey][p].clamp(min=1e-3)
                    mc, vc = fuse(gm, gv_, owner, len(ids))
                    lg = nll_logits(qm, qv_, mc, vc)
                pred = lg.argmax(1).numpy()
                per.append((np.array([ids[k] for k in pred]) == np.array([cells[i][0] for i in p])).mean() * 100)
            accs.append(np.mean(per))
        res.append(np.mean(accs))
    return float(np.mean(res))


SCEN = {'clean': ('clean', 'clean'), 'corrupt_enroll': ('occ1', 'clean'), 'hband': ('clean', 'hband'),
        'bottom': ('clean', 'bottom'), 'vband': ('clean', 'vband'), 'noise': ('clean', 'noise')}
SELECT = ['clean', 'corrupt_enroll', 'hband']                 # what the head saw during calibration


def evaluate(bb, seeds=(0, 1, 2, 3, 4)):
    torch.set_num_threads(14)
    ds, test, cells = cells_and_ds()
    folds = [test[0::2], test[1::2]]
    M = {v: torch.load(f'{OUT}/hyb_{bb}_{v}.pt').float() for v in VARIANTS}
    rows = part_rows(bb)
    E = {v: torch.stack([torch.load(f'{OUT}/hyb_energy_{v}.pt')[:, lo:hi].mean(1) for lo, hi in rows], 1)
         for v in VARIANTS}                                                          # [N, P]
    idx = {c: i for i, c in enumerate(cells)}
    res = J(RES) if os.path.exists(RES) else {}
    res[bb] = {}
    for f in range(2):
        cal, ev = folds[f], folds[1 - f]
        ebar = E['clean'][[idx[c] for c in cells if c[0] in set(cal)]].mean(0)
        Rr = {rule: {v: reliability(E[v], ebar, rule) for v in VARIANTS} for rule in RULES}
        sel = lambda kind, V: max(RULES, key=lambda rule: np.mean(
            [rank1(cells, cal, kind, M, V, Rr[rule], *SCEN[s]) for s in SELECT]))
        er = sel('energy', None)
        r = {'frozen': {s: rank1(cells, ev, 'frozen', M, None, None, *SCEN[s]) for s in SCEN},
             'energy': {s: rank1(cells, ev, 'energy', M, None, Rr[er], *SCEN[s]) for s in SCEN},
             'energy_rule': str(er), 'head': {}, 'hybrid': {}, 'hybrid_rule': {}}
        print(bb, 'fold', f, 'frozen', r['frozen'], 'energy', er, r['energy'], flush=True)
        for seed in seeds:
            h = fit_head(cells, M, cal, seed, ds, 'cpu', ['occ1', 'occ2'])
            with torch.no_grad():
                V = {v: h(M[v]) for v in VARIANTS}
            hr = sel('hybrid', V)
            r['head'][seed] = {s: rank1(cells, ev, 'head', M, V, None, *SCEN[s]) for s in SCEN}
            r['hybrid'][seed] = {s: rank1(cells, ev, 'hybrid', M, V, Rr[hr], *SCEN[s]) for s in SCEN}
            r['hybrid_rule'][seed] = str(hr)
            print(bb, 'fold', f, 'seed', seed, 'head', {k: round(x, 2) for k, x in r['head'][seed].items()},
                  'hybrid', hr, {k: round(x, 2) for k, x in r['hybrid'][seed].items()}, flush=True)
        res[bb][f'fold{f}'] = r
        save_json(res, RES)


def repro(bb):
    """Published CASIA-B protocol (nearest neighbour, all 50 test subjects) on the clean embeddings."""
    ds, test, cells = cells_and_ds()
    E = torch.load(f'{OUT}/hyb_{bb}_clean.pt').float()
    idx = {c: i for i, c in enumerate(cells)}
    gq = [ds.qidx(x) for x in LT_GALLERY]
    res = {}
    for cond in ['nm', 'bg', 'cl']:
        pq = [ds.qidx(x) for x in LT_PROBES[cond]]
        accs = []
        for pv in range(11):
            per = []
            for gv in range(11):
                if gv == pv:
                    continue
                g = [idx[(s, x, gv)] for s in test for x in gq if (s, x, gv) in idx]
                p = [idx[(s, x, pv)] for s in test for x in pq if (s, x, pv) in idx]
                pred = np.array([cells[g[j]][0] for j in dist(E[p], E[g]).argmin(1).numpy()])
                per.append((pred == np.array([cells[i][0] for i in p])).mean() * 100)
            accs.append(np.mean(per))
        res[f'published_nn>{cond}'] = float(np.mean(accs))
    print(bb, {k: round(v, 2) for k, v in res.items()}, flush=True)
    path = 'results/p5_frozen/repro.json'
    out = J(path) if os.path.exists(path) else {}
    out[bb] = res
    save_json(out, path)


def J(p):
    import json
    return json.load(open(p))


if __name__ == '__main__':
    {'embed': embed, 'eval': evaluate, 'repro': repro}[sys.argv[1]](sys.argv[2])
