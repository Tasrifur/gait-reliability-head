"""Variance-shrinkage fallback for the plug-in head on frozen CASIA-B checkpoints.

After the head predicts per-part variances s_i, they are shrunk towards one shared constant in log space,
    log s_i' = (1 - lam) log s_i + lam log s_bar,
s_bar = mean variance over all parts and clean CALIBRATION-fold cells. At lam = 1 every part of every
sample has the same variance and, by Proposition 2, the head ranks like the uniform squared distance to
the mean prototype (the frozen backbone's own metric is the mean per-part Euclidean distance, which can
rank slightly differently; 'sq' reports the uniform squared distance as a control).

lam in LAMS is selected per fold and seed on the calibration identities, by the mean rank-1 over
SELECT (clean, corrupted enrollment, horizontal band; as the hybrid rule of frozen_hybrid), and
alternatively with clothing change added ('sel_cl'). Every lam is also evaluated on the evaluation fold,
so the full curve is saved. Optionally the energy prior (Eq. hybrid) is applied after shrinkage, with the
energy rule selected in the frozen_hybrid run for the same fold and seed.

Protocol and embeddings as frozen_hybrid (2-fold cross-fit over the 50 unseen LT test subjects, 5 head
seeds, identical head fitting). Output: results/p5_plugin/frozen_shrink.json
Usage: python -m gaitlab.frozen_shrink <bb> [<bb> ...]
"""
import os
import sys

import numpy as np
import torch

from .data.casiab import LT_GALLERY, LT_PROBES, SEQS
from .eval_enrollment import GAL, PRB
from .external_eval import dist
from .frozen_hybrid import OUT, VARIANTS, cells_and_ds, part_rows, reliability
from .plugin_head import fit_head, fuse, nll_logits
from .utils import save_json

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
RES = 'results/p5_plugin/frozen_shrink.json'
C3 = ['nm', 'bg', 'cl']
# metric -> list of (gallery variant, probe variant, gallery seqs, probe seqs); value = mean over the list
METRICS = {f'clean_{c}': [('clean', 'clean', LT_GALLERY, LT_PROBES[c])] for c in C3}
METRICS['mismatch'] = [('clean', 'clean', GAL[g], PRB[p]) for g in C3 for p in C3 if g != p]
METRICS['corrupt_enroll'] = [('occ1', 'clean', LT_GALLERY, LT_PROBES[c]) for c in C3]
for _v in ['hband', 'bottom', 'vband', 'noise']:
    METRICS[_v] = [('clean', _v, LT_GALLERY, LT_PROBES[c]) for c in C3]
SELECT_KEYS = ['clean_nm', 'clean_bg', 'clean_cl', 'corrupt_enroll', 'hband']


class Matcher:
    """Full-gallery cross-view rank-1 (as frozen_hybrid.rank1), with cached index structures."""

    def __init__(self, cells, device):
        self.cells, self.dev = cells, device
        self.idx = {c: i for i, c in enumerate(cells)}
        self.cache = {}

    def pairs(self, subjects, gseqs, pseqs):
        key = (tuple(subjects), tuple(gseqs), tuple(pseqs))
        if key not in self.cache:
            idx, gq, pq, out = self.idx, [SEQS.index(x) for x in gseqs], [SEQS.index(x) for x in pseqs], []
            for pv in range(11):
                per = []
                for gv in range(11):
                    if gv == pv:
                        continue
                    ids = [s for s in subjects if any((s, x, gv) in idx for x in gq)]
                    rows = [idx[(s, x, gv)] for s in ids for x in gq if (s, x, gv) in idx]
                    owner = [ids.index(self.cells[r][0]) for r in rows]
                    p = [idx[(s, x, pv)] for s in ids for x in pq if (s, x, pv) in idx]
                    if not p:
                        continue
                    lab = [ids.index(self.cells[i][0]) for i in p]
                    t = lambda z: torch.tensor(z, device=self.dev)
                    per.append((t(rows), t(owner), len(ids), t(p), t(lab)))
                out.append(per)
            self.cache[key] = out
        return self.cache[key]

    @torch.no_grad()
    def acc(self, subjects, gseqs, pseqs, M, gkey, pkey, V=None):
        accs = []
        for per in self.pairs(subjects, gseqs, pseqs):
            a = []
            for rows, owner, n, p, lab in per:
                gm, qm = M[gkey][rows], M[pkey][p]
                if V is None:                                   # frozen: backbone metric, mean prototype
                    proto = torch.zeros(n, *gm.shape[1:], device=self.dev).index_add_(0, owner, gm)
                    proto /= torch.bincount(owner, minlength=n)[:, None, None].float()
                    lg = -dist(qm, proto)
                elif isinstance(V, str):                        # 'sq': uniform squared distance (control)
                    proto = torch.zeros(n, *gm.shape[1:], device=self.dev).index_add_(0, owner, gm)
                    proto /= torch.bincount(owner, minlength=n)[:, None, None].float()
                    lg = -((qm[:, None] - proto[None]) ** 2).sum((2, 3))
                else:
                    mc, vc = fuse(gm, V[gkey][rows], owner, n)
                    lg = nll_logits(qm, V[pkey][p], mc, vc)
                a.append((lg.argmax(1) == lab).float().mean().item() * 100)
            accs.append(np.mean(a))
        return float(np.mean(accs))

    def metrics(self, subjects, M, V=None, keys=None):
        return {k: float(np.mean([self.acc(subjects, gs, ps, M, gk, pk, V) for gk, pk, gs, ps in METRICS[k]]))
                for k in (keys or METRICS)}


def summary(m):
    """Paper-level metrics from the raw ones."""
    out = {k: v for k, v in m.items() if not k.startswith('clean_')}
    out['clean'] = float(np.mean([m[f'clean_{c}'] for c in C3]))
    out['cl'] = m['clean_cl']
    return out


def shrink(V, sbar, lam):
    return {k: torch.exp((1 - lam) * torch.log(v) + lam * np.log(sbar)) for k, v in V.items()}


def evaluate(bb, seeds=(0, 1, 2, 3, 4), device='cuda'):
    torch.set_num_threads(14)
    ds, test, cells = cells_and_ds()
    folds = [test[0::2], test[1::2]]
    M = {v: torch.load(f'{OUT}/hyb_{bb}_{v}.pt').float() for v in VARIANTS}
    E = {v: torch.stack([torch.load(f'{OUT}/hyb_energy_{v}.pt')[:, lo:hi].mean(1) for lo, hi in part_rows(bb)], 1)
         for v in VARIANTS}
    H = J('results/p5_plugin/frozen_hybrid.json')[bb]          # energy rules selected in the hybrid run
    Md = {k: v.to(device) for k, v in M.items()}
    mt = Matcher(cells, device)
    idx = {c: i for i, c in enumerate(cells)}
    res = J(RES) if os.path.exists(RES) else {}
    res[bb] = {}
    for f in range(2):
        cal, ev = folds[f], folds[1 - f]
        cal_rows = [idx[c] for c in cells if c[0] in set(cal)]
        ebar = E['clean'][cal_rows].mean(0)
        r = {'frozen': summary(mt.metrics(ev, Md)), 'sq': summary(mt.metrics(ev, Md, 'sq')), 'seeds': {}}
        print(bb, 'fold', f, 'frozen', {k: round(v, 2) for k, v in r['frozen'].items()},
              'sq', {k: round(v, 2) for k, v in r['sq'].items()}, flush=True)
        for seed in seeds:
            h = fit_head(cells, M, cal, seed, ds, 'cpu', ['occ1', 'occ2'])
            with torch.no_grad():
                V = {v: h(M[v]).to(device) for v in VARIANTS}
            sbar = float(V['clean'][cal_rows].mean())
            rule = H[f'fold{f}']['hybrid_rule'][str(seed)]
            R = {v: reliability(E[v], ebar, float(rule) if rule != 'soft' else 'soft').to(device) for v in VARIANTS}
            out = {'sbar': sbar, 'hybrid_rule': rule, 'head': {}, 'hybrid': {}}
            for kind in ['head', 'hybrid']:
                cal_m, ev_m = {}, {}
                for lam in LAMS:
                    Vl = shrink(V, sbar, lam)
                    if kind == 'hybrid':
                        Vl = {k: v / R[k].clamp(min=1e-3) for k, v in Vl.items()}
                    cal_m[lam] = mt.metrics(cal, Md, Vl, SELECT_KEYS)
                    ev_m[lam] = summary(mt.metrics(ev, Md, Vl))
                crit = lambda lam, cl: np.mean([np.mean([cal_m[lam][f'clean_{c}'] for c in C3]),
                                                cal_m[lam]['corrupt_enroll'], cal_m[lam]['hband']] +
                                               ([cal_m[lam]['clean_cl']] if cl else []))
                sel, sel_cl = max(LAMS, key=lambda l: crit(l, False)), max(LAMS, key=lambda l: crit(l, True))
                out[kind] = {'lam_sel': sel, 'lam_sel_cl': sel_cl, 'eval': {str(l): ev_m[l] for l in LAMS}}
                print(bb, 'fold', f, 'seed', seed, kind, 'lam', sel, '(+CL:', sel_cl, ')',
                      {k: round(v, 2) for k, v in ev_m[sel].items()}, flush=True)
            r['seeds'][str(seed)] = out
        res[bb][f'fold{f}'] = r
        save_json(res, RES)


def J(p):
    import json
    return json.load(open(p))


if __name__ == '__main__':
    for b in sys.argv[1:]:
        evaluate(b)
