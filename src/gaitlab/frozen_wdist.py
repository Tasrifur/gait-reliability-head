"""Precision-weighted backbone distance (paper Eq. wdist, Table frozenwdist) for the plug-in head on frozen
CASIA-B checkpoints. The head and its fitted variances are unchanged (same 2-fold cross-fit, 5 seeds, CPU head
fitting and embeddings as frozen_hybrid / frozen_shrink); only the matching score changes, to OpenGait's own
metric weighted by the head's precisions:

  enrollment  c_{n,i} = sum_k a_{k,i} mu_i(x_k),  a_{k,i} ~ s_i(x_k)^-gamma (normalised over k),
              s_i(c_n) = 1 / sum_k s_i(x_k)^-1   (fused variance, as in plugin_head.fuse)
  score       -sum_i w_i d_i / sum_i w_i,  d_i = ||mu_i(q) - c_{n,i}||_2,  w_i = (s_i(q) + s_i(c_n))^-gamma

gamma = 0 is the frozen backbone exactly (mean prototype, mean per-part Euclidean distance). gamma in GAMMAS
is selected per fold and seed on the calibration identities by the mean rank-1 over clean, corrupted
enrollment and horizontal band ('gamma_sel'), and with clothing change added ('gamma_sel_cl'). The same is
run with the energy prior applied to the variances first (s_i <- s_i / max(r_i, 1e-3), energy rule of the
frozen_hybrid run for the same fold and seed). For reference the current likelihood head (and head +
energy) is re-evaluated with the same fitted heads. Open-set metrics (as plugin_head.final_protocol: raw or
margin score, rule chosen on the calibration fold by mean OSCR) are computed for frozen and for the new
score at the selected gamma.

Output: results/p5_plugin/frozen_wdist.json.   Usage: python -m gaitlab.frozen_wdist <bb> [<bb> ...]
"""
import os
import sys

import numpy as np
import torch

from .data.casiab import LT_GALLERY, LT_PROBES, SEQS
from .eval_openset import auroc, dir_at_far, oscr
from .frozen_hybrid import OUT, VARIANTS, cells_and_ds, part_rows, reliability
from .frozen_shrink import C3, METRICS, SELECT_KEYS, Matcher, summary
from .plugin_head import fit_head
from .utils import save_json

GAMMAS = [0.0, 0.5, 1.0, 2.0]
RES = 'results/p5_plugin/frozen_wdist.json'
MODES = ['nm', 'bg', 'cl', 'nm_occ']


def wd_logits(gm, gv, owner, n, qm, qv, gamma):
    a = gv ** -gamma                                                       # [m, P]
    proto = torch.zeros(n, *gm.shape[1:], device=gm.device).index_add_(0, owner, gm * a[:, None])
    proto /= torch.zeros(n, gv.shape[1], device=gm.device).index_add_(0, owner, a)[:, None]
    vc = 1.0 / torch.zeros(n, gv.shape[1], device=gm.device).index_add_(0, owner, 1.0 / gv)
    d = torch.sqrt(((qm[:, None] - proto[None]) ** 2).sum(2).clamp(min=0))   # [q, n, P]
    w = (qv[:, None] + vc[None]) ** -gamma
    return -(w * d).sum(-1) / w.sum(-1)


class WMatcher(Matcher):
    """Matcher with the precision-weighted Euclidean score (V = (variances, gamma))."""

    @torch.no_grad()
    def acc(self, subjects, gseqs, pseqs, M, gkey, pkey, V=None):
        if not (isinstance(V, tuple)):
            return super().acc(subjects, gseqs, pseqs, M, gkey, pkey, V)
        Vd, gamma = V
        accs = []
        for per in self.pairs(subjects, gseqs, pseqs):
            a = [(wd_logits(M[gkey][rows], Vd[gkey][rows], owner, n, M[pkey][p], Vd[pkey][p], gamma)
                  .argmax(1) == lab).float().mean().item() * 100 for rows, owner, n, p, lab in per]
            accs.append(np.mean(a))
        return float(np.mean(accs))

    @torch.no_grad()
    def openset(self, subjects, M, V=None):
        """Open-set as plugin_head.evaluate_backbone (10 enrolled/unknown splits, pooled scores)."""
        idx, gq = self.idx, [SEQS.index(x) for x in LT_GALLERY]
        res = {}
        for mode in MODES:
            cond = 'nm' if mode == 'nm_occ' else mode
            pkey = 'occ2' if mode == 'nm_occ' else 'clean'
            pq = [SEQS.index(x) for x in LT_PROBES[cond]]
            rng = np.random.RandomState(2024)
            reps = []
            for _ in range(10):
                enr = set(sorted(rng.choice(subjects, len(subjects) // 2, replace=False).tolist()))
                sk, su, ck = {'raw': [], 'margin': []}, {'raw': [], 'margin': []}, []
                for gv in range(11):
                    ids = [s for s in sorted(enr) if any((s, x, gv) in idx for x in gq)]
                    rows = torch.tensor([idx[(s, x, gv)] for s in ids for x in gq if (s, x, gv) in idx],
                                        device=self.dev)
                    owner = torch.tensor([ids.index(self.cells[r][0]) for r in rows.tolist()], device=self.dev)
                    for pv in range(11):
                        if pv == gv:
                            continue
                        pc = [(s, x, pv) for s in subjects for x in pq if (s, x, pv) in idx]
                        p = torch.tensor([idx[c] for c in pc], device=self.dev)
                        gm, qm = M['clean'][rows], M[pkey][p]
                        if V is None:
                            n = len(ids)
                            proto = torch.zeros(n, *gm.shape[1:], device=self.dev).index_add_(0, owner, gm)
                            proto /= torch.bincount(owner, minlength=n)[:, None, None].float()
                            lg = -torch.sqrt(((qm[:, None] - proto[None]) ** 2).sum(2).clamp(min=0)).mean(-1)
                        else:
                            Vd, gamma = V
                            lg = wd_logits(gm, Vd['clean'][rows], owner, len(ids), qm, Vd[pkey][p], gamma)
                        t2 = lg.topk(2, 1)
                        sc = {'raw': t2.values[:, 0].cpu().numpy(),
                              'margin': (t2.values[:, 0] - t2.values[:, 1]).cpu().numpy()}
                        known = np.array([c[0] in enr for c in pc])
                        am = t2.indices[:, 0].cpu().numpy()
                        ck += [ids[a] == c[0] for c, a, k in zip(pc, am, known) if k]
                        for r in sc:
                            sk[r].append(sc[r][known]); su[r].append(sc[r][~known])
                ck = np.array(ck)
                o = {}
                for r in sk:
                    k_, u_ = np.concatenate(sk[r]), np.concatenate(su[r])
                    o[f'{r}/auroc'], o[f'{r}/oscr'] = auroc(k_, u_), oscr(k_, ck, u_)
                    o[f'{r}/dir@far1'] = dir_at_far(k_, ck, u_, 0.01)
                reps.append(o)
            for k in reps[0]:
                res[f'os_{mode}/{k}'] = float(np.mean([z[k] for z in reps]))
        return res


def open_summary(cal, ev):
    """Rule chosen on the calibration fold (best mean OSCR over modes); the t_frozen open-set metrics."""
    rule = max(['raw', 'margin'], key=lambda r: np.mean([cal[f'os_{m}/{r}/oscr'] for m in MODES]))
    return {'rule': rule, 'cl_auroc': 100 * ev[f'os_cl/{rule}/auroc'], 'cl_oscr': 100 * ev[f'os_cl/{rule}/oscr'],
            'occ_auroc': 100 * ev[f'os_nm_occ/{rule}/auroc'], 'occ_oscr': 100 * ev[f'os_nm_occ/{rule}/oscr'],
            'nm_dir1': 100 * ev[f'os_nm/{rule}/dir@far1']}


def crit(m, cl):
    return np.mean([np.mean([m[f'clean_{c}'] for c in C3]), m['corrupt_enroll'], m['hband']] +
                   ([m['clean_cl']] if cl else []))


def evaluate(bb, seeds=(0, 1, 2, 3, 4), device='cuda', open_set=True):
    torch.set_num_threads(14)
    ds, test, cells = cells_and_ds()
    folds = [test[0::2], test[1::2]]
    M = {v: torch.load(f'{OUT}/hyb_{bb}_{v}.pt').float() for v in VARIANTS}
    E = {v: torch.stack([torch.load(f'{OUT}/hyb_energy_{v}.pt')[:, lo:hi].mean(1) for lo, hi in part_rows(bb)], 1)
         for v in VARIANTS}
    H = J('results/p5_plugin/frozen_hybrid.json')[bb]
    Md = {k: v.to(device) for k, v in M.items()}
    mt = WMatcher(cells, device)
    idx = {c: i for i, c in enumerate(cells)}
    res = J(RES) if os.path.exists(RES) else {}
    res[bb] = {}
    for f in range(2):
        cal, ev = folds[f], folds[1 - f]
        ebar = E['clean'][[idx[c] for c in cells if c[0] in set(cal)]].mean(0)
        r = {'frozen': summary(mt.metrics(ev, Md)), 'seeds': {}}
        if open_set:
            r['frozen_open'] = open_summary(mt.openset(cal, Md), mt.openset(ev, Md))
        print(bb, 'fold', f, 'frozen', {k: round(v, 2) for k, v in r['frozen'].items()}, flush=True)
        for seed in seeds:
            h = fit_head(cells, M, cal, seed, ds, 'cpu', ['occ1', 'occ2'])
            with torch.no_grad():
                V = {v: h(M[v]).to(device) for v in VARIANTS}
            rule = H[f'fold{f}']['hybrid_rule'][str(seed)]
            R = {v: reliability(E[v], ebar, float(rule) if rule != 'soft' else 'soft').to(device) for v in VARIANTS}
            VE = {k: v / R[k].clamp(min=1e-3) for k, v in V.items()}
            out = {'hybrid_rule': rule,
                   'head': summary(mt.metrics(ev, Md, V)), 'head_energy': summary(mt.metrics(ev, Md, VE))}
            for name, Vs in [('wd', V), ('wd_energy', VE)]:
                cal_m = {g: mt.metrics(cal, Md, (Vs, g), SELECT_KEYS) for g in GAMMAS}
                ev_m = {g: summary(mt.metrics(ev, Md, (Vs, g))) for g in GAMMAS}
                sel = max(GAMMAS, key=lambda g: crit(cal_m[g], False))
                sel_cl = max(GAMMAS, key=lambda g: crit(cal_m[g], True))
                out[name] = {'gamma_sel': sel, 'gamma_sel_cl': sel_cl, 'eval': {str(g): ev_m[g] for g in GAMMAS}}
                if open_set:
                    out[name]['open'] = open_summary(mt.openset(cal, Md, (Vs, sel)), mt.openset(ev, Md, (Vs, sel)))
                print(bb, 'fold', f, 'seed', seed, name, 'gamma', sel, '(+CL:', sel_cl, ')',
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
