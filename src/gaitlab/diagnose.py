"""Phase 2 diagnostics for DPP-Net checkpoints (CASIA-B test subjects).

  A. attention statistics: mean weight per part, normalised entropy, effective #parts
  B. per-part solo rank-1 (only part i used in the distance) + test-time-uniform
  C. episode dependence: weights of the same query under different galleries
  D. margin decomposition: per-part share of (d_runnerup - d_pred) vs weight share

All analyses use the standard full-gallery cross-view protocol (gallery NM-01..04
at one view, probe at a different view) unless noted.

Usage: python -m gaitlab.diagnose --run results/v0_dppnet_lt_cv --out results/phase2_diag/cv
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from .data.casiab import CasiaB, LT_GALLERY, LT_PROBES, VIEWS
from .models.dppnet import DPPNet
from .utils import mean_ci, save_json

PARTS = ['head', 'shoulders', 'upper torso', 'lower torso', 'pelvis', 'thighs', 'calves', 'feet']


@torch.no_grad()
def embed_all(model, ds, subjects, device):
    """Embeddings for every valid test cell: dict (s,q,v) -> [P,D] tensor."""
    cells = [(s, q, v) for s in subjects for q in range(10) for v in range(11) if ds.valid[s, q, v]]
    out = {}
    for i in range(0, len(cells), 512):
        chunk = cells[i:i + 512]
        X = torch.from_numpy(np.stack([ds.get(*c) for c in chunk])).to(device)
        E = model.encoder(X)
        for c, e in zip(chunk, E):
            out[c] = e
    return out


def gallery(emb, ds, subjects, gv):
    gq = [ds.qidx(q) for q in LT_GALLERY]
    ids = [s for s in subjects if any((s, q, gv) in emb for q in gq)]
    proto = torch.stack([torch.stack([emb[(s, q, gv)] for q in gq if (s, q, gv) in emb]).mean(0)
                         for s in ids])
    return ids, proto


@torch.no_grad()
def analyse(model, ds, subjects, device, rng):
    emb = embed_all(model, ds, subjects, device)
    P = model.P
    res = {}
    for cond in ['nm', 'bg', 'cl']:
        pq = [ds.qidx(q) for q in LT_PROBES[cond]]
        W, solo_hits, uni_hits, full_hits, n = [], np.zeros(P), 0, 0, 0
        shares_w, shares_m, top_mismatch = [], [], []
        for gv in range(11):
            ids, proto = gallery(emb, ds, subjects, gv)
            for pv in range(11):
                if pv == gv:
                    continue
                qc = [(s, q, pv) for s in ids for q in pq if (s, q, pv) in emb]
                if not qc:
                    continue
                Q = torch.stack([emb[c] for c in qc])
                lab = torch.tensor([ids.index(c[0]) for c in qc], device=device)
                w = model.part_weights(Q, proto)                       # [Nq, P]
                d = ((Q[:, None] - proto[None]) ** 2).sum(-1)          # [Nq, N, P]
                logits = -(d * w[:, None]).sum(-1)
                full_hits += (logits.argmax(1) == lab).sum().item()
                uni_hits += ((-d.mean(-1)).argmax(1) == lab).sum().item()
                for i in range(P):
                    solo_hits[i] += ((-d[..., i]).argmax(1) == lab).sum().item()
                n += len(qc)
                W.append(w.cpu())
                # D. margin decomposition between predicted and runner-up
                top2 = logits.topk(2, dim=1).indices
                ar = torch.arange(len(qc), device=device)
                contrib = w * (d[ar, top2[:, 1]] - d[ar, top2[:, 0]])  # [Nq, P], sums to margin
                shares_m.append(contrib.cpu())   # ratio of sums computed below (robust to tiny margins)
                shares_w.append(w.cpu())
                top_mismatch.append((contrib.argmax(1) != w.argmax(1)).float().cpu())
        W = torch.cat(W)
        H = -(W * W.clamp(min=1e-12).log()).sum(1) / np.log(P)
        sm, sw = torch.cat(shares_m), torch.cat(shares_w)
        res[cond] = {
            'rank1_dynamic': 100 * full_hits / n,
            'rank1_test_time_uniform': 100 * uni_hits / n,
            'rank1_solo_part': {PARTS[i]: 100 * solo_hits[i] / n for i in range(P)},
            'mean_weight': {PARTS[i]: float(W[:, i].mean()) for i in range(P)},
            'entropy_norm_mean': float(H.mean()),
            'effective_parts_mean': float(torch.exp(H * np.log(P)).mean()),
            'frac_weight_top4': float(W.sort(1, descending=True).values[:, :4].sum(1).mean()),
            'margin_share_mean': {PARTS[i]: float(sm[:, i].sum() / sm.sum()) for i in range(P)},
            'frac_top_weight_part_not_top_margin_part': float(torch.cat(top_mismatch).mean()),
            'frac_parts_with_negative_margin_contrib': float((sm < 0).float().mean()),
            'frac_queries_with_any_part_against': float((sm < 0).any(1).float().mean()),
            'n_queries': n,
        }

    # C. episode dependence: same queries, different gallery compositions
    pq = [ds.qidx(q) for q in LT_PROBES['cl']]
    gv, pv = VIEWS.index('090'), VIEWS.index('054')
    ids, proto_full = gallery(emb, ds, subjects, gv)
    qc = [(s, q, pv) for s in ids for q in pq if (s, q, pv) in emb]
    Q = torch.stack([emb[c] for c in qc])
    w_full = model.part_weights(Q, proto_full)
    dep = {}
    for N in [1, 5, 20]:
        l1, flip = [], []
        for _ in range(20):
            ws = []
            for j, c in enumerate(qc):
                t = ids.index(c[0])
                others = [k for k in range(len(ids)) if k != t]
                pick = [t] + list(rng.choice(others, N - 1, replace=False)) if N > 1 else [t]
                ws.append(model.part_weights(Q[j:j + 1], proto_full[pick])[0])
            ws = torch.stack(ws)
            l1.append((ws - w_full).abs().sum(1).mean().item())
            flip.append((ws.argmax(1) != w_full.argmax(1)).float().mean().item())
        dep[f'N={N}_vs_N={len(ids)}'] = {'mean_L1': float(np.mean(l1)),
                                         'top_part_changes': float(np.mean(flip))}
    res['episode_dependence_cl_090to054'] = dep
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    ds = CasiaB('data/cache/casiab_gei.npz')
    per_seed = []
    for ck in sorted(glob.glob(os.path.join(args.run, 'seed*', 'best.pt'))):
        cfg = json.load(open(os.path.join(os.path.dirname(ck), 'config.json')))
        split = ds.lt_split(cfg['data'].get('n_val', 10))
        m = DPPNet(attention=cfg['model']['attention']).to(args.device).eval()
        m.load_state_dict(torch.load(ck, weights_only=True))
        print('analysing', ck, flush=True)
        per_seed.append(analyse(m, ds, split['test'], args.device, np.random.RandomState(0)))
    os.makedirs(args.out, exist_ok=True)
    save_json(per_seed, os.path.join(args.out, 'per_seed.json'))

    # seed-aggregated summary of scalar fields
    def agg(path):
        vals = []
        for r in per_seed:
            x = r
            for k in path:
                x = x[k]
            vals.append(x)
        return mean_ci(vals)
    summ = {}
    for cond in ['nm', 'bg', 'cl']:
        s = {}
        for k in ['rank1_dynamic', 'rank1_test_time_uniform', 'entropy_norm_mean',
                  'effective_parts_mean', 'frac_weight_top4',
                  'frac_top_weight_part_not_top_margin_part',
                  'frac_parts_with_negative_margin_contrib', 'frac_queries_with_any_part_against']:
            s[k] = agg([cond, k])
        for k in ['rank1_solo_part', 'mean_weight', 'margin_share_mean']:
            s[k] = {p: agg([cond, k, p])['mean'] for p in PARTS}
        summ[cond] = s
    summ['episode_dependence'] = {k: {m: agg(['episode_dependence_cl_090to054', k, m])
                                      for m in ['mean_L1', 'top_part_changes']}
                                  for k in per_seed[0]['episode_dependence_cl_090to054']}
    save_json(summ, os.path.join(args.out, 'summary.json'))
    for c in ['nm', 'bg', 'cl']:
        s = summ[c]
        print(f"== {c.upper()}: rank1 dyn {s['rank1_dynamic']['mean']:.2f} | test-time-uniform "
              f"{s['rank1_test_time_uniform']['mean']:.2f} | eff.parts {s['effective_parts_mean']['mean']:.2f}"
              f" | top-weight!=top-margin {100*s['frac_top_weight_part_not_top_margin_part']['mean']:.1f}%"
              f" | queries w/ a part against {100*s['frac_queries_with_any_part_against']['mean']:.1f}%")
        print('   part        weight  margin-share  solo-rank1')
        for p_ in PARTS:
            print(f"   {p_:<11} {s['mean_weight'][p_]:.3f}   {s['margin_share_mean'][p_]:+.3f}       {s['rank1_solo_part'][p_]:.2f}")
    for k, v in summ['episode_dependence'].items():
        print(f"episode dependence {k}: mean L1 {v['mean_L1']['mean']:.3f}, top part changes {100*v['top_part_changes']['mean']:.1f}%")


if __name__ == '__main__':
    main()
