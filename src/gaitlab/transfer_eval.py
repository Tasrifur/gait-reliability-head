"""Phase 6: retraining-free transfer of CASIA-B-trained GEI models to CASIA-E and OU-MVLP.

Official OpenGait protocols, applied to our GEI caches:
  CASIA-E : test subjects 201-1014 (814). Gallery = scene1 NM seq 1-2, heights H+L
            (4 GEIs per subject-view). Probes = scene2 / scene3 / scene3 's' variant,
            NM / BG / CL, seq 1-2, both heights. View 270 excluded (13 views).
            Seq 3/4 (only at 045/135; a second camera) are excluded, as in the
            official protocol.
  OU-MVLP : test = even subject ids (5,154). Gallery seq 01, probe seq 00, 14 views.
Metric: full-gallery rank-1 with prototype enrollment; 'cv' = identical view
excluded (per-probe-view mean over gallery views, then mean over probe views,
as OpenGait's de_diag); 'sv' = identical view only.
Models (5 seeds each, trained only on CASIA-B): L0 original DPP-Net (cross-view
trained), B1, model of record (B1 + occlusion-aware N1/N2, lambda=1).
Also: CASIA-E 045/135 diagnostic (probes from seq 3/4 vs seq 1/2).

Usage: python -m gaitlab.transfer_eval
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from .eval_enrollment import load_pm
from .models.dppnet import DPPNet
from .models.protonet import ProtoNet
from .utils import mean_ci, save_json

OUT = 'results/p6_transfer'
CE_VIEWS = ['000', '015', '030', '045', '060', '075', '090', '105', '120', '135', '150', '165', '180']
OU_VIEWS = ['000', '015', '030', '045', '060', '075', '090', '180', '195', '210', '225', '240', '255', '270']


class Model:
    def __init__(self, kind, seed, device):
        self.kind, self.device = kind, device
        if kind == 'L0':
            self.m = DPPNet(attention='residual').to(device).eval()
            self.m.load_state_dict(torch.load(f'results/v0_dppnet_lt_cv/seed{seed}/best.pt', weights_only=True))
        elif kind == 'B1':
            self.m = ProtoNet('v0bn').to(device).eval()
            self.m.load_state_dict(torch.load(f'results/p3_bn_aux/seed{seed}/best.pt', weights_only=True))
        else:
            self.m = load_pm('results/p4_2stage_occ', seed, device)
            self.m.fusion, self.m.lam = 'precision', 1.0

    @torch.no_grad()
    def embed(self, X):
        out_m, out_v = [], []
        for i in range(0, len(X), 1024):
            x = torch.from_numpy(np.asarray(X[i:i + 1024], dtype=np.float32)[:, None] / 255.0).to(self.device)
            if self.kind == 'N':
                mu, var = self.m.embed(x)
                out_m.append(mu); out_v.append(var)
            else:
                out_m.append(self.m.encoder(x))
        return torch.cat(out_m), (torch.cat(out_v) if out_v else None)

    @torch.no_grad()
    def logits(self, gm, gv, owner, n, qm, qv, chunk=512):
        """Memory-safe scoring: per-strip squared distances via ||q||^2+||c||^2-2q.c."""
        if self.kind == 'N':
            c, vc = self.m.fuse(gm, gv, owner, n)
        else:
            c = torch.zeros(n, *gm.shape[1:], device=gm.device).index_add_(0, owner, gm)
            c /= torch.bincount(owner, minlength=n).clamp(min=1)[:, None, None].float()
        c2 = (c ** 2).sum(-1)                                            # [n, P]
        out = []
        for i in range(0, len(qm), chunk):
            q = qm[i:i + chunk]
            d = ((q ** 2).sum(-1)[:, None] + c2[None] - 2 * torch.einsum('qpd,npd->qnp', q, c)).clamp(min=0)
            if self.kind == 'N':
                sv = qv[i:i + chunk][:, None] + vc[None]
                out.append(-0.5 * (d / sv + self.m.lam * self.m.D * torch.log(sv)).sum(-1))
            elif self.kind == 'L0':
                w = self.m.part_weights(q, c)
                out.append(-(d * w[:, None]).sum(-1))
            else:
                out.append(-self.m.log_scale.exp() * d.mean(-1))
        return torch.cat(out)


def evaluate(model, X, gal, probes, views):
    """gal: DataFrame of gallery rows; probes: {cond: DataFrame}. Returns per (cond, pv, gv) acc."""
    idx = np.unique(np.concatenate([gal.index.values] + [p.index.values for p in probes.values()]))
    pos = {r: i for i, r in enumerate(idx)}
    M, V = model.embed(X[idx])
    acc = {c: np.full((len(views), len(views)), np.nan) for c in probes}
    for gi, gvw in enumerate(views):
        g = gal[gal.angle == gvw]
        subs = sorted(g.subject.unique())
        sid = {s: i for i, s in enumerate(subs)}
        rows = torch.tensor([pos[r] for r in g.index], device=model.device)
        owner = torch.tensor([sid[s] for s in g.subject], device=model.device)
        gm, gvv = M[rows], (V[rows] if V is not None else None)
        for c, P in probes.items():
            for pi, pvw in enumerate(views):
                p = P[(P.angle == pvw) & P.subject.isin(sid)]
                if len(p) == 0:
                    continue
                pr = torch.tensor([pos[r] for r in p.index], device=model.device)
                lg = model.logits(gm, gvv, owner, len(subs), M[pr], V[pr] if V is not None else None)
                pred = lg.argmax(1).cpu().numpy()
                acc[c][pi, gi] = (np.array([subs[k] for k in pred]) == p.subject.values).mean() * 100
    return acc


def summarize(acc):
    out = {}
    for c, a in acc.items():
        off = np.where(np.eye(len(a), dtype=bool), np.nan, a)
        out[f'{c}/cv'] = float(np.nanmean(np.nanmean(off, 1)))          # de_diag: per probe view, then mean
        out[f'{c}/sv'] = float(np.nanmean(np.diag(a)))
        out[f'{c}/cv_per_probe_view'] = np.nanmean(off, 1).tolist()
        out[f'{c}/sv_per_view'] = np.diag(a).tolist()
    return out


def main():
    device = 'cuda'
    os.makedirs(OUT, exist_ok=True)
    # ---------------- CASIA-E ----------------
    XE = np.load('data/cache/casiae_gei.npy', mmap_mode='r')
    me = pd.read_csv('data/cache/casiae_meta.csv', dtype=str).fillna('')   # empty variant -> '' (not NaN)
    me['seq'] = me.seq.astype(int)
    te = me[(me.subject.astype(int) >= 201) & me.angle.isin(CE_VIEWS)]
    gal_e = te[(te.scene == 'scene1') & (te.variant == '') & (te.cond == 'nm') & te.seq.isin([1, 2])]
    prb_e = {f'{sc}{"s" if v else ""}-{c}': te[(te.scene == sc) & (te.variant == v) & (te.cond == c) & te.seq.isin([1, 2])]
             for sc, v in [('scene2', ''), ('scene3', ''), ('scene3', 's')] for c in ['nm', 'bg', 'cl']}
    # 045/135 diagnostic: probes from the other camera (seq 3/4), scene2 NM
    diag_e = {'scene2-nm-seq34': te[(te.scene == 'scene2') & (te.variant == '') & (te.cond == 'nm') & te.seq.isin([3, 4])]}
    # ---------------- OU-MVLP ----------------
    XO = np.load('data/cache/oumvlp_gei.npy', mmap_mode='r')
    mo = pd.read_csv('data/cache/oumvlp_meta.csv', dtype=str)
    mo['seq'] = mo.seq.astype(int)
    to = mo[mo.subject.astype(int) % 2 == 0]
    gal_o, prb_o = to[to.seq == 1], {'nm': to[to.seq == 0]}
    print('CASIA-E gallery', len(gal_e), 'subjects', gal_e.subject.nunique(), '| probes',
          {k: len(v) for k, v in prb_e.items()}, '| OU gallery', len(gal_o), 'probes', len(prb_o['nm']), flush=True)

    res = {}
    for kind in ['L0', 'B1', 'N']:
        res[kind] = {}
        for seed in range(5):
            m = Model(kind, seed, device)
            r = {'casiae': summarize(evaluate(m, XE, gal_e, prb_e, CE_VIEWS)),
                 'casiae_seq34': summarize(evaluate(m, XE, gal_e, diag_e, CE_VIEWS)),
                 'oumvlp': summarize(evaluate(m, XO, gal_o, prb_o, OU_VIEWS))}
            res[kind][seed] = r
            ce = {c: np.mean([r['casiae'][f'{s}-{c}/cv'] for s in ['scene2', 'scene3', 'scene3s']]) for c in ['nm', 'bg', 'cl']}
            print(kind, seed, 'CASIA-E cv', {k: round(v, 2) for k, v in ce.items()},
                  'OU cv %.2f sv %.2f' % (r['oumvlp']['nm/cv'], r['oumvlp']['nm/sv']), flush=True)
            save_json(res, f'{OUT}/per_seed.json')


if __name__ == '__main__':
    main()
