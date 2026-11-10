"""Phase 4d: faithfulness of per-strip explanations.

Every model scores a query against prototype n as a SUM over strips, so the
margin between the predicted identity p and the runner-up r decomposes exactly:
    margin = logit_p - logit_r = sum_i c_i
    old attention (DPP-Net v0) : c_i = w_i (d_i(r) - d_i(p))      [w_i = learned weight]
    B1 (uniform)               : c_i = (d_i(r) - d_i(p)) / P
    N1+N2                      : c_i = nll_i(r) - nll_i(p)          [Gaussian per-strip NLL]

Explanations compared (each ranks strips by importance for ONE decision):
    contrib : c_i                         (margin decomposition)
    weight  : w_i (attention) or 1/(s_q,i + s_p,i) (N1+N2): "where the model looks"
    random  : random order (baseline);  anti : reverse of contrib (sanity)

Deletion test (input level, not tautological): blank the image rows of the
top-k ranked strips in the PROBE GEI, re-run the full model with the same
gallery, and record whether the top-1 decision flips. A faithful explanation
flips decisions faster than random. Deletion AUC = mean flip rate over k=1..4;
faithfulness gain = AUC(expl) - AUC(random).

Protocol: CASIA-B LT test, cross-view; each probe (NM-05/06, BG, CL) is matched
against all test subjects' NM-01..04 at one random other view.

Usage: python -m gaitlab.interpret --out results/p4d_interpret
"""
import argparse
import os

import numpy as np
import torch

from .data.casiab import CasiaB, LT_GALLERY, LT_PROBES
from .eval_enrollment import load_pm
from .models.dppnet import DPPNet
from .models.protonet import ProtoNet
from .utils import mean_ci, save_json

KMAX = 4


class Wrap:
    """Uniform interface: proto(support cells) and per-strip logits for queries."""

    def __init__(self, kind, seed, device):
        self.kind, self.device = kind, device
        if kind == 'attention':
            self.m = DPPNet(attention='residual').to(device).eval()
            self.m.load_state_dict(torch.load(f'results/v0_dppnet_lt_cv/seed{seed}/best.pt', weights_only=True))
        elif kind == 'b1':
            self.m = ProtoNet('v0bn').to(device).eval()
            self.m.load_state_dict(torch.load(f'results/p3_bn_aux/seed{seed}/best.pt', weights_only=True))
        else:
            self.m = load_pm('results/p4_2stage_occ', seed, device)
            self.m.fusion, self.m.lam = 'precision', 1.0

    @torch.no_grad()
    def prototypes(self, X, owner, n):
        if self.kind == 'n1n2':
            mu, var = self.m.embed(X)
            return self.m.fuse(mu, var, owner, n)
        E = self.m.encoder(X)
        c = torch.zeros(n, E.shape[1], E.shape[2], device=X.device).index_add_(0, owner, E)
        return c / torch.bincount(owner, minlength=n).clamp(min=1)[:, None, None].float(), None

    @torch.no_grad()
    def part_logits(self, Q, proto):
        """Returns per-strip logit contributions [Nq, N, P] (sum over P = logits)
        and the 'weight' explanation [Nq, N, P]."""
        mc, vc = proto
        if self.kind == 'n1n2':
            mq, vq = self.m.embed(Q)
            nll = self.m.part_nll(mq, vq, mc, vc)
            w = 1.0 / (vq[:, None] + vc[None])
            return -nll, w
        q = self.m.encoder(Q)
        d = ((q[:, None] - mc[None]) ** 2).sum(-1)
        if self.kind == 'attention':
            w = self.m.part_weights(q, mc)[:, None].expand_as(d)
        else:
            w = torch.full_like(d, 1.0 / d.shape[-1])
        return -d * w, w


def blank(X, strips):
    X = X.clone()
    for i, ss in enumerate(strips):
        for j in ss:
            X[i, :, 8 * j:8 * j + 8, :] = 0
    return X


@torch.no_grad()
def run(model, ds, test, cond, device, rng, n_probe=600):
    gq = [ds.qidx(q) for q in LT_GALLERY]
    pq = [ds.qidx(q) for q in LT_PROBES[cond]]
    probes = [(s, q, v) for s in test for q in pq for v in range(11) if ds.valid[s, q, v]]
    probes = [probes[i] for i in rng.choice(len(probes), min(n_probe, len(probes)), replace=False)]
    gvs = [int(rng.choice([g for g in range(11) if g != c[2]])) for c in probes]
    flips = {e: np.zeros(KMAX) for e in ['contrib', 'weight', 'random', 'anti']}
    stats = {'top2_share': [], 'neg_frac': [], 'weight_vs_contrib_top1_agree': []}
    n_total = 0
    for gv in sorted(set(gvs)):
        idx = [i for i, g in enumerate(gvs) if g == gv]
        ids = [s for s in test if any(ds.valid[s, q, gv] for q in gq)]
        sup = [(s, q, gv) for s in ids for q in gq if ds.valid[s, q, gv]]
        X = torch.from_numpy(np.stack([ds.get(*c) for c in sup])).to(device)
        owner = torch.tensor([ids.index(c[0]) for c in sup], device=device)
        proto = model.prototypes(X, owner, len(ids))
        qc = [probes[i] for i in idx if probes[i][0] in ids]
        if not qc:
            continue
        Q = torch.from_numpy(np.stack([ds.get(*c) for c in qc])).to(device)
        pl, w = model.part_logits(Q, proto)
        logits = pl.sum(-1)
        top2 = logits.topk(2, 1).indices
        ar = torch.arange(len(qc), device=device)
        p, r = top2[:, 0], top2[:, 1]
        contrib = pl[ar, p] - pl[ar, r]                                # [Nq, P], sums to margin
        margin = contrib.sum(1)
        stats['top2_share'] += (contrib.sort(1, descending=True).values[:, :2].sum(1)
                                / margin.clamp(min=1e-8)).clamp(max=5).tolist()
        stats['neg_frac'] += (contrib < 0).float().mean(1).tolist()
        wexp = w[ar, p]
        stats['weight_vs_contrib_top1_agree'] += (wexp.argmax(1) == contrib.argmax(1)).float().tolist()
        orders = {'contrib': contrib.argsort(1, descending=True),
                  'weight': wexp.argsort(1, descending=True),
                  'anti': contrib.argsort(1),
                  'random': torch.stack([torch.randperm(contrib.shape[1], device=device)
                                         for _ in range(len(qc))])}
        for e, o in orders.items():
            o = o.cpu().numpy()
            for k in range(1, KMAX + 1):
                Qb = blank(Q, o[:, :k])
                pred = model.part_logits(Qb, proto)[0].sum(-1).argmax(1)
                flips[e][k - 1] += (pred != p).sum().item()
        n_total += len(qc)
    out = {f'flip_{e}': (flips[e] / n_total).tolist() for e in flips}
    out.update({f'auc_{e}': float((flips[e] / n_total).mean()) for e in flips})
    out.update({k: float(np.mean(v)) for k, v in stats.items()})
    out['n'] = n_total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/p4d_interpret')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    per = {}
    for seed in a.seeds:
        torch.manual_seed(seed)
        per[seed] = {}
        for kind in ['attention', 'b1', 'n1n2']:
            m = Wrap(kind, seed, a.device)
            per[seed][kind] = {c: run(m, ds, test, c, a.device, np.random.RandomState(seed))
                               for c in ['nm', 'bg', 'cl']}
            r = per[seed][kind]
            print(f'seed {seed} {kind:9s} ' + ' | '.join(
                f"{c}: gain(contrib) {r[c]['auc_contrib'] - r[c]['auc_random']:+.3f} "
                f"gain(weight) {r[c]['auc_weight'] - r[c]['auc_random']:+.3f}" for c in r), flush=True)
        save_json(per, os.path.join(a.out, 'per_seed.json'))
    summ = {}
    for kind in ['attention', 'b1', 'n1n2']:
        summ[kind] = {}
        for c in ['nm', 'bg', 'cl']:
            s = {}
            for e in ['contrib', 'weight', 'anti']:
                s[f'gain_{e}'] = mean_ci([per[k][kind][c][f'auc_{e}'] - per[k][kind][c]['auc_random'] for k in a.seeds])
            for key in ['auc_random', 'top2_share', 'neg_frac', 'weight_vs_contrib_top1_agree']:
                s[key] = mean_ci([per[k][kind][c][key] for k in a.seeds])
            s['flip_curves'] = {e: np.mean([per[k][kind][c][f'flip_{e}'] for k in a.seeds], 0).tolist()
                                for e in ['contrib', 'weight', 'random', 'anti']}
            summ[kind][c] = s
    save_json(summ, os.path.join(a.out, 'summary.json'))
    print('\n| model | cond | flip AUC random | gain contrib | gain weight | gain anti | top-2 share | neg. strips | weight=contrib top-1 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for kind in summ:
        for c in ['nm', 'bg', 'cl']:
            s = summ[kind][c]
            print(f"| {kind} | {c} | {s['auc_random']['mean']:.3f} | {s['gain_contrib']['mean']:+.3f} ± {s['gain_contrib']['ci95']:.3f}"
                  f" | {s['gain_weight']['mean']:+.3f} ± {s['gain_weight']['ci95']:.3f} | {s['gain_anti']['mean']:+.3f}"
                  f" | {s['top2_share']['mean']:.2f} | {s['neg_frac']['mean']:.2f} | {s['weight_vs_contrib_top1_agree']['mean']:.2f} |")


if __name__ == '__main__':
    main()
