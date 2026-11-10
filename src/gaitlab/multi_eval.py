"""Multi-dataset evaluation: B1 vs B1 + reliability head calibrated on held-out ids.

Datasets / official test protocols (full gallery, cross-view = identical view excluded):
  casiab : LT test 075-124; gallery NM-01..04; probes NM-05/06, BG, CL (11 views)
  casiae : test 201-1014; gallery scene1 NM seq1-2 H+L; probes scenes 2/3/3s x NM/BG/CL (13 views)
  oumvlp : test even ids; gallery seq 01; probe seq 00 (14 views)
Scenarios (all on test ids):
  fgcv            clean gallery/probes
  mismatch        gallery from BG or CL instead of NM (casiab, casiae)
  corrupt_enroll  every gallery GEI band-occluded with p=0.5 (as Phase 4b S3)
  occ_<type>      every probe corrupted: hband | vband | bottom | noise (occlusions.py);
                  the head was calibrated with horizontal bands only
  open-set        half of the test ids enrolled; rule chosen on the CALIBRATION ids
                  (B1: raw|margin, head: raw|margin|lr; best mean OSCR over modes)

Usage: python -m gaitlab.multi_eval --dataset casiab|casiae|oumvlp
"""
import argparse
import os

import numpy as np
import torch

from .calibrate import load_dataset
from .data.casiab import LT_GALLERY, LT_PROBES
from .eval_enrollment import occlude
from .eval_openset import FARS, auroc, dir_at_far, oscr
from .models.probnet import ProbProtoNet
from .occlusions import TYPES, corrupt
from .utils import save_json

B1_DIR = {'casiab': 'results/p3_bn_aux', 'casiae': 'results/md_casiae_b1', 'oumvlp': 'results/md_oumvlp_b1'}
HEAD_DIR = {d: f'results/md_{d}_head' for d in B1_DIR}
OS_SPLITS = {'casiab': 10, 'casiae': 5, 'oumvlp': 2}


def protocol(name, ds):
    if name == 'casiab':
        return (LT_GALLERY, dict(LT_PROBES),
                {'bg': (['bg-01'], {'nm': ['nm-05', 'nm-06'], 'bg': ['bg-02'], 'cl': ['cl-02']}),
                 'cl': (['cl-01'], {'nm': ['nm-05', 'nm-06'], 'bg': ['bg-02'], 'cl': ['cl-02']})})
    if name == 'casiae':
        sc1 = lambda c: [q for q in ds.seqs if f'-scene1-{c}-' in q]
        return ds.seq_set('gallery'), ds.probe_sets(), {'bg': (sc1('bg'), ds.probe_sets()),
                                                        'cl': (sc1('cl'), ds.probe_sets())}
    return ['01'], {'nm': ['00']}, {}


class Scorer:
    def __init__(self, name, seed, with_head, device):
        self.pm = ProbProtoNet('v0bn').to(device).eval()
        b1 = torch.load(f'{B1_DIR[name]}/seed{seed}/best.pt', weights_only=True)
        self.pm.cnn.load_state_dict({k[4:]: v for k, v in b1.items() if k.startswith('cnn.')})
        if with_head:
            self.pm.load_state_dict(torch.load(f'{HEAD_DIR[name]}/seed{seed}/var_head.pt', weights_only=True),
                                    strict=False)
        self.head, self.device = with_head, device

    @torch.no_grad()
    def embed(self, X):
        M, V = [], []
        for i in range(0, len(X), 1024):
            mu, var = self.pm.embed(torch.from_numpy(X[i:i + 1024]).to(self.device))
            M.append(mu); V.append(var)
        return torch.cat(M), torch.cat(V)

    @torch.no_grad()
    def logits(self, gm, gv, owner, n, qm, qv, chunk=512, pop=False):
        if self.head:
            c, vc = self.pm.fuse(gm, gv, owner, n)
        else:
            c = torch.zeros(n, *gm.shape[1:], device=gm.device).index_add_(0, owner, gm)
            c /= torch.bincount(owner, minlength=n).clamp(min=1)[:, None, None].float()
        c2 = (c ** 2).sum(-1)
        out, lpop = [], []
        if pop and self.head:
            mpop = c.mean(0, keepdim=True)
            vpop = (vc.mean(0) + ((c - mpop) ** 2).sum(-1).mean(0) / c.shape[-1])[None]
        for i in range(0, len(qm), chunk):
            q = qm[i:i + chunk]
            d = ((q ** 2).sum(-1)[:, None] + c2[None] - 2 * torch.einsum('qpd,npd->qnp', q, c)).clamp(min=0)
            if self.head:
                s = qv[i:i + chunk][:, None] + vc[None]
                out.append(-0.5 * (d / s + self.pm.D * torch.log(s)).sum(-1))
                if pop:
                    sp = qv[i:i + chunk] + vpop
                    dp = ((q - mpop) ** 2).sum(-1)
                    lpop.append(-0.5 * (dp / sp + self.pm.D * torch.log(sp)).sum(-1))
            else:
                out.append(-d.mean(-1))
        return torch.cat(out), (torch.cat(lpop) if lpop else None)


class Cells:
    """Embeds the needed cells once per variant; lookup by (s,q,v)."""

    def __init__(self, ds, scorer, subjects, gal_seqs, probe_seqs):
        gq = sorted({ds.qidx(q) for q in gal_seqs})
        pq = sorted({ds.qidx(q) for q in probe_seqs})
        self.g = [(s, q, v) for s in subjects for q in gq for v in range(len(ds.views)) if ds.valid[s, q, v]]
        self.p = [(s, q, v) for s in subjects for q in pq for v in range(len(ds.views)) if ds.valid[s, q, v]]
        self.gi = {c: i for i, c in enumerate(self.g)}
        self.pi = {c: i for i, c in enumerate(self.p)}
        # streamed in chunks: CASIA-E has ~378k probe GEIs (5.8 GB per float32 copy)
        def emb(cells, fn, chunk=4096):
            M, V = [], []
            for i in range(0, len(cells), chunk):
                X = np.stack([fn(ds.get(*c), c) for c in cells[i:i + chunk]])
                m, v = scorer.embed(X)
                M.append(m.half()); V.append(v.half())     # fp16 storage; scoring is done in fp32
            return torch.cat(M), torch.cat(V)
        self.emb = {('g', 'clean'): emb(self.g, lambda x, c: x),
                    ('p', 'clean'): emb(self.p, lambda x, c: x),
                    ('g', 'corrupt'): emb(self.g, lambda x, c: occlude(x, c, 1, 0.5))}
        for t in TYPES:
            self.emb[('p', t)] = emb(self.p, lambda x, c, t=t: corrupt(x, c, t))


def rank1(ds, sc, cells, subjects, gseqs, pseqs, gvar='clean', pvar='clean'):
    gq, pq = [ds.qidx(q) for q in gseqs], [ds.qidx(q) for q in pseqs]
    GM, GV = cells.emb[('g', gvar)]; PM, PV = cells.emb[('p', pvar)]
    nv = len(ds.views)
    accs = []
    for pv in range(nv):
        per = []
        for gv in range(nv):
            if gv == pv:
                continue
            rows = [cells.gi[(s, q, gv)] for s in subjects for q in gq if (s, q, gv) in cells.gi]
            if not rows:
                continue
            owners = [cells.g[r][0] for r in rows]
            ids = sorted(set(owners)); sid = {s: i for i, s in enumerate(ids)}
            pr = [cells.pi[(s, q, pv)] for s in ids for q in pq if (s, q, pv) in cells.pi]
            if not pr:
                continue
            owner = torch.tensor([sid[o] for o in owners], device=sc.device)
            r = torch.tensor(rows, device=sc.device); p = torch.tensor(pr, device=sc.device)
            lg, _ = sc.logits(GM[r].float(), GV[r].float(), owner, len(ids), PM[p].float(), PV[p].float())
            pred = lg.argmax(1).cpu().numpy()
            per.append((np.array([ids[k] for k in pred]) == np.array([cells.p[i][0] for i in pr])).mean() * 100)
        if per:
            accs.append(np.mean(per))
    return float(np.mean(accs))


def openset(ds, sc, cells, subjects, gseqs, probes, splits, seed=2024):
    """Returns {mode: {rule/metric: value}} for modes = probe conds + 'occ_hband'."""
    rng = np.random.RandomState(seed)
    rules = ['raw', 'margin'] + (['lr'] if sc.head else [])
    gq = [ds.qidx(q) for q in gseqs]
    modes = {**{c: (qs, 'clean') for c, qs in probes.items()}, 'occ_hband': (probes['nm'], 'hband')}
    acc = {m: [] for m in modes}
    for _ in range(splits):
        enr = set(rng.choice(subjects, len(subjects) // 2, replace=False).tolist())
        for m, (qs, var) in modes.items():
            pq = [ds.qidx(q) for q in qs]
            GM, GV = cells.emb[('g', 'clean')]; PM, PV = cells.emb[('p', var)]
            sk = {r: [] for r in rules}; su = {r: [] for r in rules}; ck = []
            for gv in range(len(ds.views)):
                rows = [cells.gi[(s, q, gv)] for s in sorted(enr) for q in gq if (s, q, gv) in cells.gi]
                if not rows:
                    continue
                owners = [cells.g[r][0] for r in rows]; ids = sorted(set(owners)); sid = {s: i for i, s in enumerate(ids)}
                owner = torch.tensor([sid[o] for o in owners], device=sc.device)
                r = torch.tensor(rows, device=sc.device)
                for pv in range(len(ds.views)):
                    if pv == gv:
                        continue
                    pr = [cells.pi[(s, q, pv)] for s in subjects for q in pq if (s, q, pv) in cells.pi]
                    if not pr:
                        continue
                    p = torch.tensor(pr, device=sc.device)
                    lg, lpop = sc.logits(GM[r].float(), GV[r].float(), owner, len(ids), PM[p].float(), PV[p].float(), pop=True)
                    t2 = lg.topk(2, 1)
                    sco = {'raw': t2.values[:, 0], 'margin': t2.values[:, 0] - t2.values[:, 1]}
                    if sc.head:
                        sco['lr'] = t2.values[:, 0] - lpop
                    known = np.array([cells.p[i][0] in enr for i in pr])
                    am = t2.indices[:, 0].cpu().numpy()
                    ck += [ids[a] == cells.p[i][0] for i, a, k in zip(pr, am, known) if k]
                    for x in rules:
                        v = sco[x].cpu().numpy(); sk[x].append(v[known]); su[x].append(v[~known])
            ck = np.array(ck); o = {}
            for x in rules:
                k_, u_ = np.concatenate(sk[x]), np.concatenate(su[x])
                o[f'{x}/auroc'], o[f'{x}/oscr'] = auroc(k_, u_), oscr(k_, ck, u_)
                for f in FARS:
                    o[f'{x}/dir@far{int(f * 100)}'] = dir_at_far(k_, ck, u_, f)
            acc[m].append(o)
    return {m: {k: float(np.mean([z[k] for z in v])) for k in v[0]} for m, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    ds, sp, _, _, _ = load_dataset(a.dataset)
    gseqs, probes, mism = protocol(a.dataset, ds)
    all_probe = sorted({q for qs in probes.values() for q in qs} |
                       {q for _, pr in mism.values() for qs in pr.values() for q in qs})
    all_gal = sorted(set(gseqs) | {q for g, _ in mism.values() for q in g})
    out_path = f'results/md_eval_{a.dataset}.json'
    res = {}
    for seed in a.seeds:
        res[seed] = {}
        for kind in ['b1', 'head']:
            sc = Scorer(a.dataset, seed, kind == 'head', a.device)
            test = Cells(ds, sc, sp['test'], all_gal, all_probe)
            cal = Cells(ds, sc, sp['val'], all_gal, all_probe)
            r = {}
            for c, qs in probes.items():
                r[f'fgcv>{c}'] = rank1(ds, sc, test, sp['test'], gseqs, qs)
                r[f'corrupt_enroll>{c}'] = rank1(ds, sc, test, sp['test'], gseqs, qs, gvar='corrupt')
                for t in TYPES:
                    r[f'occ_{t}>{c}'] = rank1(ds, sc, test, sp['test'], gseqs, qs, pvar=t)
            for g, (gq, pr) in mism.items():
                for c, qs in pr.items():
                    if c != g:
                        r[f'mismatch_{g}>{c}'] = rank1(ds, sc, test, sp['test'], gq, qs)
            os_cal = openset(ds, sc, cal, sp['val'], gseqs, probes, max(OS_SPLITS[a.dataset], 4))
            rules = ['raw', 'margin'] + (['lr'] if kind == 'head' else [])
            rule = max(rules, key=lambda x: np.mean([os_cal[m][f'{x}/oscr'] for m in os_cal]))
            os_test = openset(ds, sc, test, sp['test'], gseqs, probes, OS_SPLITS[a.dataset])
            r['openset_rule'] = rule
            for m, d in os_test.items():
                for k, v in d.items():
                    if k.startswith(rule + '/'):
                        r[f'os_{m}/{k.split("/", 1)[1]}'] = v
            res[seed][kind] = r
            print(a.dataset, seed, kind, 'rule', rule, '| fgcv',
                  round(np.mean([r[f'fgcv>{c}'] for c in probes]), 2), flush=True)
            save_json(res, out_path)


if __name__ == '__main__':
    main()
