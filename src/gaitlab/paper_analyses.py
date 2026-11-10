"""Extra analyses for the manuscript figures (CASIA-B, B1 vs B1 + held-out-calibrated head).

Writes results/paper_analyses.json with:
  per_view      : per-probe-view cross-view rank-1 (NM/BG/CL clean, NM hband-occluded)
  enroll_sweep  : rank-1 vs fraction p of band-occluded enrollment samples
  probe_sweep   : rank-1 vs height h of a horizontal band on every probe
  kshot         : rank-1 vs number of enrollment samples K (clean and p=0.5 corrupted)
  dirfar        : DIR as a function of FAR (log grid), seed 0, NM/CL clean + NM occluded
  reliability   : binned predicted per-part variance vs observed squared discrepancy
  timing        : parameters, enrollment time per identity, matching time per probe
Usage: python -m gaitlab.paper_analyses
"""
import time

import numpy as np
import torch

from .calibrate import load_dataset
from .data.casiab import LT_GALLERY, LT_PROBES
from .eval_enrollment import band_for, occlude
from .multi_eval import Cells, Scorer, protocol
from .utils import save_json

C3 = ['nm', 'bg', 'cl']
SEEDS = [0, 1, 2, 3, 4]


def band_h(img, cell, h):
    """Horizontal band of fixed height h at the deterministic position of the cell."""
    if h == 0:
        return img
    rng = np.random.RandomState((hash(cell) + 2) % (2 ** 31))
    rng.rand(); rng.randint(6, 17)
    t = rng.randint(0, 64 - h + 1)
    x = img.copy(); x[..., t:t + h, :] = 0
    return x


def add_variant(ds, sc, cells, side, name, fn, chunk=4096):
    cl = cells.g if side == 'g' else cells.p
    M, V = [], []
    for i in range(0, len(cl), chunk):
        X = np.stack([fn(ds.get(*c), c) for c in cl[i:i + chunk]])
        m, v = sc.embed(X); M.append(m.half()); V.append(v.half())
    cells.emb[(side, name)] = (torch.cat(M), torch.cat(V))


@torch.no_grad()
def per_view(ds, sc, cells, subjects, gseqs, pseqs, gvar='clean', pvar='clean', kmax=None):
    """Per-probe-view cross-view rank-1. kmax limits the enrollment samples per identity."""
    gq, pq = [ds.qidx(q) for q in gseqs], [ds.qidx(q) for q in pseqs]
    GM, GV = cells.emb[('g', gvar)]; PM, PV = cells.emb[('p', pvar)]
    out = []
    for pv in range(11):
        per = []
        for gv in range(11):
            if gv == pv:
                continue
            rows = []
            for s in subjects:
                r = [cells.gi[(s, q, gv)] for q in gq if (s, q, gv) in cells.gi]
                rows += r[:kmax] if kmax else r
            owners = [cells.g[r][0] for r in rows]; ids = sorted(set(owners)); sid = {s: i for i, s in enumerate(ids)}
            pr = [cells.pi[(s, q, pv)] for s in ids for q in pq if (s, q, pv) in cells.pi]
            owner = torch.tensor([sid[o] for o in owners], device=sc.device)
            r = torch.tensor(rows, device=sc.device); p = torch.tensor(pr, device=sc.device)
            lg, _ = sc.logits(GM[r].float(), GV[r].float(), owner, len(ids), PM[p].float(), PV[p].float())
            pred = lg.argmax(1).cpu().numpy()
            per.append((np.array([ids[k] for k in pred]) == np.array([cells.p[i][0] for i in pr])).mean() * 100)
        out.append(float(np.mean(per)))
    return out


@torch.no_grad()
def dirfar_curve(ds, sc, cells, subjects, gseqs, pseqs, pvar, rule, splits=3):
    gq, pq = [ds.qidx(q) for q in gseqs], [ds.qidx(q) for q in pseqs]
    GM, GV = cells.emb[('g', 'clean')]; PM, PV = cells.emb[('p', pvar)]
    rng = np.random.RandomState(2024)
    fars = np.logspace(-3, 0, 25)
    curves = []
    for _ in range(splits):
        enr = set(rng.choice(subjects, len(subjects) // 2, replace=False).tolist())
        sk, su, ck = [], [], []
        for gv in range(11):
            rows = [cells.gi[(s, q, gv)] for s in sorted(enr) for q in gq if (s, q, gv) in cells.gi]
            owners = [cells.g[r][0] for r in rows]; ids = sorted(set(owners)); sid = {s: i for i, s in enumerate(ids)}
            owner = torch.tensor([sid[o] for o in owners], device=sc.device); r = torch.tensor(rows, device=sc.device)
            for pv in range(11):
                if pv == gv:
                    continue
                pr = [cells.pi[(s, q, pv)] for s in subjects for q in pq if (s, q, pv) in cells.pi]
                p = torch.tensor(pr, device=sc.device)
                lg, lpop = sc.logits(GM[r].float(), GV[r].float(), owner, len(ids), PM[p].float(), PV[p].float(), pop=True)
                t2 = lg.topk(2, 1)
                s_ = {'raw': t2.values[:, 0], 'margin': t2.values[:, 0] - t2.values[:, 1]}
                if sc.head:
                    s_['lr'] = t2.values[:, 0] - lpop
                v = s_[rule].cpu().numpy(); known = np.array([cells.p[i][0] in enr for i in pr])
                am = t2.indices[:, 0].cpu().numpy()
                sk.append(v[known]); su.append(v[~known])
                ck += [ids[a] == cells.p[i][0] for i, a, k in zip(pr, am, known) if k]
        sk, su, ck = np.concatenate(sk), np.concatenate(su), np.array(ck)
        curves.append([float(((sk > np.quantile(su, 1 - f, method='higher')) & ck).mean()) for f in fars])
    return fars.tolist(), np.mean(curves, 0).tolist()


@torch.no_grad()
def reliability(ds, sc, cells, subjects, nbins=10):
    """Bin probe part variances; per bin, mean observed squared discrepancy / D to own prototype."""
    gq = [ds.qidx(q) for q in LT_GALLERY]; pq = [ds.qidx(q) for q in sum(LT_PROBES.values(), [])]
    GM, GV = cells.emb[('g', 'clean')]
    pred, obs = [], []
    for pvar in ['clean', 'hband']:
        PM, PV = cells.emb[('p', pvar)]
        for gv in range(11):
            for s in subjects:
                rows = [cells.gi[(s, q, gv)] for q in gq if (s, q, gv) in cells.gi]
                if not rows:
                    continue
                r = torch.tensor(rows, device=sc.device)
                mc, vc = sc.pm.fuse(GM[r].float(), GV[r].float(), torch.zeros(len(rows), dtype=torch.long, device=sc.device), 1)
                pr = [cells.pi[(s, q, pv)] for q in pq for pv in range(11) if pv != gv and (s, q, pv) in cells.pi]
                p = torch.tensor(pr, device=sc.device)
                d = ((PM[p].float() - mc) ** 2).sum(-1) / sc.pm.D
                pred.append((PV[p].float() + vc).cpu().numpy().ravel()); obs.append(d.cpu().numpy().ravel())
    pred, obs = np.concatenate(pred), np.concatenate(obs)
    edges = np.quantile(pred, np.linspace(0, 1, nbins + 1))
    b = np.clip(np.searchsorted(edges, pred, side='right') - 1, 0, nbins - 1)
    return {'pred': [float(pred[b == k].mean()) for k in range(nbins)],
            'obs': [float(obs[b == k].mean()) for k in range(nbins)],
            'spearman': float(__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(pred, obs)[0])}


def timing(ds, device):
    from .external_eval import build
    out = {}
    x4 = torch.rand(4, 1, 64, 64, device=device); q = torch.rand(1, 1, 64, 64, device=device)
    for name, head in [('B1', False), ('B1+head', True)]:
        sc = Scorer('casiab', 0, head, device)
        n = sum(p.numel() for p in sc.pm.cnn.parameters()) + (sum(p.numel() for p in sc.pm.var_head.parameters()) + sc.pm.P if head else 0)
        gm, gv = sc.pm.embed(torch.rand(200, 1, 64, 64, device=device)); owner = torch.arange(50, device=device).repeat_interleave(4)
        for _ in range(10):
            sc.pm.embed(x4)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(200):
            m, v = sc.pm.embed(x4); sc.pm.fuse(m, v, torch.zeros(4, dtype=torch.long, device=device), 1)
        torch.cuda.synchronize(); te = (time.perf_counter() - t0) / 200 * 1e3
        t0 = time.perf_counter()
        for _ in range(200):
            m, v = sc.pm.embed(q); sc.logits(gm, gv, owner, 50, m, v)
        torch.cuda.synchronize(); tm = (time.perf_counter() - t0) / 200 * 1e3
        out[name] = {'params': int(n), 'enroll_ms_per_identity_K4': te, 'match_ms_per_probe_50way': tm}
    return out


def main():
    device = 'cuda'
    ds, sp, _, _, _ = load_dataset('casiab')
    gseqs, probes, _ = protocol('casiab', ds)
    test = sp['test']
    allp = sorted({q for v in probes.values() for q in v})
    R = {k: {} for k in ['per_view', 'enroll_sweep', 'probe_sweep', 'kshot', 'dirfar', 'reliability']}
    for seed in SEEDS:
        for kind in ['b1', 'head']:
            sc = Scorer('casiab', seed, kind == 'head', device)
            cells = Cells(ds, sc, test, gseqs, allp)
            for p in [0.25, 0.75, 1.0]:
                add_variant(ds, sc, cells, 'g', f'corrupt{p}', lambda x, c, p=p: occlude(x, c, 1, p))
            for h in [8, 16, 24, 32]:
                add_variant(ds, sc, cells, 'p', f'h{h}', lambda x, c, h=h: band_h(x, c, h))
            key = f'{kind}/{seed}'
            R['per_view'][key] = {c: per_view(ds, sc, cells, test, gseqs, probes[c]) for c in C3}
            R['per_view'][key]['nm_occ'] = per_view(ds, sc, cells, test, gseqs, probes['nm'], pvar='hband')
            R['enroll_sweep'][key] = {str(p): float(np.mean([np.mean(per_view(ds, sc, cells, test, gseqs, probes[c],
                                                                               gvar={0.0: 'clean', 0.5: 'corrupt'}.get(p, f'corrupt{p}')))
                                                             for c in C3])) for p in [0.0, 0.25, 0.5, 0.75, 1.0]}
            R['probe_sweep'][key] = {str(h): float(np.mean([np.mean(per_view(ds, sc, cells, test, gseqs, probes[c],
                                                                              pvar='clean' if h == 0 else f'h{h}'))
                                                            for c in C3])) for h in [0, 8, 16, 24, 32]}
            R['kshot'][key] = {f'{g}/K{k}': float(np.mean([np.mean(per_view(ds, sc, cells, test, gseqs, probes[c],
                                                                            gvar=g, kmax=k)) for c in C3]))
                               for g in ['clean', 'corrupt'] for k in [1, 2, 3, 4]}
            if seed == 0:
                rule = 'lr' if kind == 'head' else 'raw'
                R['dirfar'][kind] = {m: dirfar_curve(ds, sc, cells, test, gseqs, probes[c], v, rule)
                                     for m, (c, v) in {'nm': ('nm', 'clean'), 'cl': ('cl', 'clean'),
                                                       'nm_occ': ('nm', 'hband')}.items()}
            if kind == 'head':
                R['reliability'][str(seed)] = reliability(ds, sc, cells, test)
            print('done', key, flush=True)
            save_json(R, 'results/paper_analyses.json')
    R['timing'] = timing(ds, device)
    save_json(R, 'results/paper_analyses.json')
    print(R['timing'])


if __name__ == '__main__':
    main()
