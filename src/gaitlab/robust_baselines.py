"""Simple robustness baselines vs the reliability head (same B1 backbone, same cells, same protocol).

Question: is the head's robustness gain more than what a cheap, training-free heuristic gives?
Scorers (all on the frozen B1 of each seed; protocol = multi_eval full-gallery cross-view):
  b1           mean prototype, mean squared strip distance            (reference, as in multi_eval)
  median       coordinate-wise median prototype over the K enrollment GEIs (robust enrollment)
  energy       silhouette-energy strip reliability, no training: r_i = clip(e_i / e_bar_i, 0, 1),
               e_i = mean intensity of the 8 input rows under strip i, e_bar_i = mean over clean
               calibration GEIs. Enrollment: r-weighted mean per strip; matching:
               sum_i w_i d_i / sum_i w_i with w_i = r_q,i * max_k r_k,i.
               Variant (soft | hard tau in {0.25, 0.5, 0.75}) chosen on the CALIBRATION ids
               (clean + hband + corrupt_enroll, i.e. only what the head saw).
  head_global  the calibrated head with its variances pooled over strips (one scalar per GEI,
               PFE-style per-sample uncertainty): isolates the value of PER-PART reliability.
  head         ours (N1 + N2), as in multi_eval.
  hybrid       head with the energy reliability as a prior on its variances: s_i <- s_i / max(r_i, 1e-3)
               (an empty strip gets ~zero precision in enrollment and matching). No training; the
               energy variant is selected on the calibration ids separately from 'energy'.
Scenarios: fgcv (clean), corrupt_enroll (K>1 galleries only), occ_{hband,vband,bottom,noise}.
Corrupted images are cached once per dataset (data/cache/robust_<ds>.npz-style memmaps) and
reused by every seed.

Usage: python -m gaitlab.robust_baselines --dataset casiab|casiae|oumvlp [--seeds ...]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .calibrate import load_dataset
from .eval_enrollment import occlude
from .models.probnet import ProbProtoNet
from .multi_eval import B1_DIR, HEAD_DIR, protocol
from .occlusions import TYPES, corrupt
from .utils import save_json

SCORERS = ['b1', 'median', 'energy', 'head_global', 'head', 'hybrid']
RULED = ['energy', 'hybrid']                     # scorers that use the energy reliability
ENERGY_RULES = ['soft', 0.25, 0.5, 0.75]
P = 8


def cell_lists(ds, subjects, gal_seqs, probe_seqs):
    gq = sorted({ds.qidx(q) for q in gal_seqs})
    pq = sorted({ds.qidx(q) for q in probe_seqs})
    nv = len(ds.views)
    g = [(s, q, v) for s in subjects for q in gq for v in range(nv) if ds.valid[s, q, v]]
    p = [(s, q, v) for s in subjects for q in pq for v in range(nv) if ds.valid[s, q, v]]
    return g, p


def image_cache(ds, name, tag, cells, variants):
    """uint8 memmaps [N,64,64] per variant, built once (corruption is slow pure-python)."""
    out = {}
    for var in variants:
        path = f'data/cache/robust_{name}_{tag}_{var}.npy'
        if os.path.exists(path) and np.load(path, mmap_mode='r').shape[0] == len(cells):
            out[var] = np.load(path, mmap_mode='r')
            continue
        t0 = time.time()
        arr = np.lib.format.open_memmap(path + '.tmp.npy', mode='w+', dtype=np.uint8,
                                        shape=(len(cells), 64, 64))
        for i, c in enumerate(cells):
            x = ds.get(*c)
            if var == 'corrupt':
                x = occlude(x, c, 1, 0.5)
            elif var != 'clean':
                x = corrupt(x, c, var)
            arr[i] = np.round(x[0] * 255).astype(np.uint8)
        arr.flush(); del arr
        os.replace(path + '.tmp.npy', path)
        print(f'  cached {name}/{tag}/{var}: {len(cells)} GEIs in {time.time() - t0:.0f}s', flush=True)
        out[var] = np.load(path, mmap_mode='r')
    return out


def load_model(name, seed):
    pm = ProbProtoNet('v0bn').eval()
    b1 = torch.load(f'{B1_DIR[name]}/seed{seed}/best.pt', map_location='cpu', weights_only=True)
    pm.cnn.load_state_dict({k[4:]: v for k, v in b1.items() if k.startswith('cnn.')})
    pm.load_state_dict(torch.load(f'{HEAD_DIR[name]}/seed{seed}/var_head.pt', map_location='cpu',
                                  weights_only=True), strict=False)
    return pm


@torch.no_grad()
def embed(pm, imgs, chunk=2048):
    """uint8 [N,64,64] -> mu fp16 [N,P,D], var fp32 [N,P], strip energy fp32 [N,P]."""
    M, V, E = [], [], []
    for i in range(0, len(imgs), chunk):
        x = torch.from_numpy(np.array(imgs[i:i + chunk])).float().div_(255)[:, None]   # copy out of the read-only memmap
        mu, var = pm.embed(x)
        M.append(mu.half()); V.append(var); E.append(x.view(len(x), P, -1).mean(-1))
    return torch.cat(M), torch.cat(V), torch.cat(E)


class Embeddings:
    """Gallery variants embedded up front; probe variants embedded on demand, ONE held at a time
    (CASIA-E has ~378k probe GEIs; all five variants at once do not fit in 16 GB)."""

    def __init__(self, pm, gal_imgs, probe_imgs, evict=True):
        self.pm, self.pi, self.evict = pm, probe_imgs, evict
        self.d = {('g', v): embed(pm, x) for v, x in gal_imgs.items()}
        self.cur = None

    def __contains__(self, key):
        return key in self.d or (key[0] == 'p' and key[1] in self.pi)

    def __getitem__(self, key):
        if key not in self.d:
            if self.evict and self.cur is not None:
                del self.d[self.cur]
            self.d[key], self.cur = embed(self.pm, self.pi[key[1]]), key
        return self.d[key]


def reliability(e, ebar, rule):
    r = (e / ebar).clamp(0, 1)
    return r if rule == 'soft' else (r > rule).float()


def logits(kind, D, gm, gv, gr, owner, n, qm, qv, qr, chunk=512):
    """Score queries against the n prototypes built from gallery rows (owner -> 0..n-1)."""
    if kind in ('head', 'head_global', 'hybrid'):
        if kind == 'hybrid':
            gv, qv = gv / gr.clamp(min=1e-3), qv / qr.clamp(min=1e-3)
        if kind == 'head_global':
            gv, qv = gv.mean(-1, keepdim=True).expand_as(gv), qv.mean(-1, keepdim=True).expand_as(qv)
        prec = 1.0 / gv
        psum = torch.zeros(n, P).index_add_(0, owner, prec)
        c = torch.zeros(n, P, D).index_add_(0, owner, gm * prec[..., None]) / psum[..., None]
        vc = 1.0 / psum
    elif kind == 'median':
        cnt = torch.bincount(owner, minlength=n)
        k = int(cnt.max())
        slot = torch.zeros_like(owner)
        order = torch.argsort(owner, stable=True)
        start = torch.cumsum(cnt, 0) - cnt
        slot[order] = torch.arange(len(owner)) - start[owner[order]]
        pad = torch.full((n, k, P, D), float('nan'))
        pad[owner, slot] = gm
        c = pad.nanquantile(0.5, dim=1)                 # true median (mean of the middle two for even K)
    elif kind == 'energy':
        w = torch.zeros(n, P).index_add_(0, owner, gr)
        c = torch.zeros(n, P, D).index_add_(0, owner, gm * gr[..., None]) / w.clamp(min=1e-6)[..., None]
        rc = torch.zeros(n, P).index_reduce_(0, owner, gr, 'amax', include_self=False)
    else:
        cnt = torch.bincount(owner, minlength=n).clamp(min=1).float()
        c = torch.zeros(n, P, D).index_add_(0, owner, gm) / cnt[:, None, None]
    c2 = (c ** 2).sum(-1)
    out = []
    for i in range(0, len(qm), chunk):
        q = qm[i:i + chunk]
        d = ((q ** 2).sum(-1)[:, None] + c2[None] - 2 * torch.einsum('qpd,npd->qnp', q, c)).clamp(min=0)
        if kind in ('head', 'head_global', 'hybrid'):
            s = qv[i:i + chunk][:, None] + vc[None]
            out.append(-0.5 * (d / s + D * torch.log(s)).sum(-1))
        elif kind == 'energy':
            w = qr[i:i + chunk][:, None] * rc[None]
            out.append(-(w * d).sum(-1) / w.sum(-1).clamp(min=1e-6))
        else:
            out.append(-d.mean(-1))
    return torch.cat(out)


def rank1(ds, kind, D, G, Q, gcells, pcells, subjects, gseqs, pseqs):
    """G/Q = (mu, var, rel) for the chosen gallery / probe variant. Mean over probe views of
    the mean over gallery views != probe view (identical-view excluded), as in multi_eval.
    gcells / pcells: int arrays [N,3] of (subject, seq, view)."""
    gq, pq = [ds.qidx(q) for q in gseqs], [ds.qidx(q) for q in pseqs]
    gok = np.isin(gcells[:, 1], gq) & np.isin(gcells[:, 0], subjects)
    pok = np.isin(pcells[:, 1], pq) & np.isin(pcells[:, 0], subjects)
    nv = len(ds.views)
    accs = []
    for pv in range(nv):
        per = []
        for gv in range(nv):
            rows = np.where(gok & (gcells[:, 2] == gv))[0]
            if gv == pv or not len(rows):
                continue
            ids, owner = np.unique(gcells[rows, 0], return_inverse=True)
            pr = np.where(pok & (pcells[:, 2] == pv) & np.isin(pcells[:, 0], ids))[0]
            if not len(pr):
                continue
            r, p = torch.from_numpy(rows), torch.from_numpy(pr)
            lg = logits(kind, D, G[0][r].float(), G[1][r], G[2][r], torch.from_numpy(owner).long(), len(ids),
                        Q[0][p].float(), Q[1][p], Q[2][p])
            pred = ids[lg.argmax(1).numpy()]
            per.append((pred == pcells[pr, 0]).mean() * 100)
        if per:
            accs.append(np.mean(per))
    return float(np.mean(accs))


def evaluate(ds, kinds, D, emb, gcells, pcells, subjects, gseqs, probes, k_gt1, rules=None, ebar=None):
    """emb[(side, variant)] = (mu, var, energy); rules = {kind: energy rule} for RULED kinds.
    Returns {scenario>cond: {kind: acc}}."""
    def pack(side, var, kind):
        mu, v, e = emb[(side, var)]
        return mu, v, (reliability(e, ebar, rules[kind]) if kind in RULED else e)
    scen = [('fgcv', 'clean', 'clean')] + ([('corrupt_enroll', 'corrupt', 'clean')] if k_gt1 else []) + \
           [(f'occ_{t}', 'clean', t) for t in TYPES if ('p', t) in emb]
    ga, pa = np.asarray(gcells), np.asarray(pcells)
    res = {}
    for name, gvar, pvar in scen:
        for c, qs in probes.items():
            res[f'{name}>{c}'] = {k: rank1(ds, k, D, pack('g', gvar, k), pack('p', pvar, k),
                                           ga, pa, subjects, gseqs, qs) for k in kinds}
        if len(kinds) > 1:
            print(f'    {name} done ({time.strftime("%H:%M")})', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--threads', type=int, default=os.cpu_count())
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    seeds = a.seeds or sorted(int(d[4:]) for d in os.listdir(HEAD_DIR[a.dataset]) if d.startswith('seed'))
    ds, sp, _, _, _ = load_dataset(a.dataset)
    gseqs, probes, _ = protocol(a.dataset, ds)
    all_probe = sorted({q for qs in probes.values() for q in qs})
    k_gt1 = a.dataset != 'oumvlp'
    t0 = time.time()
    cells, imgs = {}, {}
    for split in ['test', 'val']:
        g, p = cell_lists(ds, sp[split], gseqs, all_probe)
        cells[split] = (g, p)
        imgs[split] = (image_cache(ds, a.dataset, f'{split}_g', g, ['clean'] + (['corrupt'] if k_gt1 else [])),
                       image_cache(ds, a.dataset, f'{split}_p', p,
                                   ['clean'] + (['hband'] if split == 'val' else TYPES)))
    print(f'{a.dataset}: test {len(cells["test"][0])} gallery / {len(cells["test"][1])} probe GEIs, '
          f'images ready in {time.time() - t0:.0f}s', flush=True)
    ds.gei = None                                   # images now come from the memmaps (frees 2.6 GB on CASIA-E)

    out_path = f'results/robust_baselines_{a.dataset}.json'
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}   # resume: finished seeds are kept
    res = {int(k): v for k, v in res.items()}
    for seed in seeds:
        if seed in res:
            print(f'  seed {seed}: already done, skipped', flush=True)
            continue
        t1 = time.time()
        pm = load_model(a.dataset, seed)
        emb = {'test': Embeddings(pm, *imgs['test']),
               'val': Embeddings(pm, *imgs['val'], evict=False)}      # calibration set is small; reused 8x
        ebar = emb['val'][('g', 'clean')][2].mean(0).clamp(min=1e-6)
        print(f'  seed {seed}: galleries embedded in {time.time() - t1:.0f}s', flush=True)
        # energy rule selection per RULED scorer on the calibration ids (clean + hband + corrupt_enroll only)
        gv_, pv_ = cells['val']
        sel, rules = {}, {}
        for kind in RULED:
            sel[kind] = {}
            for rule in ENERGY_RULES:
                r = evaluate(ds, [kind], pm.D, emb['val'], gv_, pv_, sp['val'], gseqs, probes, k_gt1,
                             {kind: rule}, ebar)
                sel[kind][str(rule)] = float(np.mean([v[kind] for v in r.values()]))
            rules[kind] = ENERGY_RULES[int(np.argmax([sel[kind][str(x)] for x in ENERGY_RULES]))]
        gt, pt = cells['test']
        r = evaluate(ds, SCORERS, pm.D, emb['test'], gt, pt, sp['test'], gseqs, probes, k_gt1, rules, ebar)
        res[seed] = {'energy_rule': {k: str(v) for k, v in rules.items()}, 'energy_rule_cal': sel, 'acc': r}
        save_json(res, out_path)
        fg = lambda k: np.mean([v[k] for s, v in r.items() if s.startswith('fgcv')])
        print(f'  seed {seed}: rules {rules}, fgcv ' + ' '.join(f'{k} {fg(k):.2f}' for k in SCORERS) +
              f' ({time.time() - t1:.0f}s)', flush=True)
        del emb
    print('saved', out_path)


if __name__ == '__main__':
    main()
