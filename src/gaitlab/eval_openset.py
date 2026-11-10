"""Phase 4c: open-set (watch-list) evaluation, N4.

Protocol (CASIA-B LT test set, 50 subjects, cross-view):
  * For each of R random splits: 25 subjects are ENROLLED (gallery NM-01..04),
    the other 25 are UNKNOWN (never enrolled).
  * For every (gallery view gv, probe view pv != gv): the gallery is the enrolled
    subjects at gv; probes are ALL 50 subjects' probe sequences at pv
    (NM-05/06, BG-01/02 or CL-01/02; plus NM with every probe band-occluded).
  * Each probe gets a top-1 identity and a score. Scores from all view pairs are
    POOLED, so a single system-wide threshold is evaluated (realistic deployment).

Scores:
  b1        : max over enrolled ids of -mean_i ||q_i - c_i||^2
  n1n2      : max over enrolled ids of the per-strip Gaussian log-likelihood
              (N1 distance with lam=1, N2 prototypes)
Metrics (known = enrolled probes, unknown = non-enrolled probes):
  AUROC      known-vs-unknown separation by score
  DIR@FAR    fraction of known probes accepted (score >= t) AND correctly
             identified, at the t where FAR (unknowns accepted) = 1%, 5%, 10%
  OSCR       area under the DIR-vs-FAR curve (FAR in [0,1])
  closed-set rank-1 on known probes (sanity)

Usage: python -m gaitlab.eval_openset --stage2 results/p4_2stage results/p4_2stage_occ
"""
import argparse
import glob
import os

import numpy as np
import torch

from .data.casiab import CasiaB, LT_GALLERY, LT_PROBES
from .eval_enrollment import load_pm, occlude
from .utils import mean_ci, save_json

FARS = [0.01, 0.05, 0.10]


# ---------------------------------------------------------------- metrics ----
def auroc(s_known, s_unknown):
    """P(score_known > score_unknown) with ties counted half (Mann-Whitney)."""
    s = np.concatenate([s_known, s_unknown])
    ranks = np.empty(len(s))
    order = s.argsort(kind='mergesort')
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(s_known), len(s_unknown)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def dir_at_far(s_known, correct, s_unknown, far):
    """Threshold = (1-far) quantile of unknown scores (accept if score > t)."""
    t = np.quantile(s_unknown, 1 - far, method='higher') if far > 0 else np.inf
    return float(((s_known > t) & correct).mean())


def oscr(s_known, correct, s_unknown):
    """Area under the curve CCR(t) vs FPR(t), sweeping t over all scores."""
    t = np.unique(np.concatenate([s_known, s_unknown]))[::-1]
    sk = np.sort(s_known[correct])[::-1]
    su = np.sort(s_unknown)[::-1]
    ccr = np.searchsorted(-sk, -t, side='right') / len(s_known)
    fpr = np.searchsorted(-su, -t, side='right') / len(s_unknown)
    ccr = np.concatenate([[0], ccr, [correct.mean()]])
    fpr = np.concatenate([[0], fpr, [1.0]])
    return float(np.trapezoid(ccr, fpr)) if hasattr(np, 'trapezoid') else float(np.trapz(ccr, fpr))


# ---------------------------------------------------------------- scoring ----
@torch.no_grad()
def embed_cells(pm, ds, cells, device, occ_salt=None, occ_p=0.0):
    mus, vars_ = [], []
    for i in range(0, len(cells), 512):
        ch = cells[i:i + 512]
        X = np.stack([occlude(ds.get(*c), c, occ_salt, occ_p) if occ_salt else ds.get(*c) for c in ch])
        mu, var = pm.embed(torch.from_numpy(X).to(device))
        mus.append(mu); vars_.append(var)
    return torch.cat(mus), torch.cat(vars_)


@torch.no_grad()
def score(pm, kind, mg, vg, owner, n, mq, vq):
    """Returns ({rule: score per probe}, argmax). Rules:
      raw    top-1 logit
      margin top-1 minus top-2 logit
      tnorm  (top-1 - mean of the other logits) / std of the other logits
      lr     (n1n2 only) top-1 log-likelihood minus log-likelihood under a
             population Gaussian (mean of enrolled prototypes; per-strip
             variance = query var + mean prototype var + between-identity var)
    """
    if kind == 'b1':
        c = torch.zeros(n, pm.P, pm.D, device=mg.device).index_add_(0, owner, mg)
        c /= torch.bincount(owner, minlength=n).clamp(min=1)[:, None, None].float()
        logits = -((mq[:, None] - c[None]) ** 2).sum(-1).mean(-1)
    else:
        pm.fusion, pm.lam = 'precision', 1.0
        mc, vc = pm.fuse(mg, vg, owner, n)
        logits = pm.logits_from_embed(mq, vq, mc, vc)
    top2 = logits.topk(2, dim=1)
    s1, a = top2.values[:, 0], top2.indices[:, 0]
    others = logits.clone()
    others[torch.arange(len(a)), a] = float('nan')
    mu_o = torch.nanmean(others, 1)
    sd_o = torch.sqrt(torch.nanmean((others - mu_o[:, None]) ** 2, 1)).clamp(min=1e-6)
    rules = {'raw': s1, 'margin': s1 - top2.values[:, 1], 'tnorm': (s1 - mu_o) / sd_o}
    if kind != 'b1':
        mpop = mc.mean(0, keepdim=True)                                     # [1,P,D]
        vbetween = ((mc - mpop) ** 2).sum(-1).mean(0) / pm.D                # [P]
        vpop = (vc.mean(0) + vbetween)[None]                                # [1,P]
        ll_pop = pm.logits_from_embed(mq, vq, mpop, vpop)[:, 0]
        rules['lr'] = s1 - ll_pop
    return {k: v.cpu().numpy() for k, v in rules.items()}, a.cpu().numpy()


def run(pm, kind, ds, test, device, repeats, probe_mode):
    cond = 'nm' if probe_mode == 'nm_occ' else probe_mode
    gq = [ds.qidx(q) for q in LT_GALLERY]
    pq = [ds.qidx(q) for q in LT_PROBES[cond]]
    gcells = [(s, q, v) for s in test for q in gq for v in range(11) if ds.valid[s, q, v]]
    pcells = [(s, q, v) for s in test for q in pq for v in range(11) if ds.valid[s, q, v]]
    MG, VG = embed_cells(pm, ds, gcells, device)
    MP, VP = embed_cells(pm, ds, pcells, device, *((2, 1.0) if probe_mode == 'nm_occ' else (None, 0)))
    gidx = {c: i for i, c in enumerate(gcells)}
    pidx = {c: i for i, c in enumerate(pcells)}
    rng = np.random.RandomState(2024)
    out = []
    for _ in range(repeats):
        enrolled = sorted(rng.choice(test, len(test) // 2, replace=False).tolist())
        enr = set(enrolled)
        sk, ck, su = {}, [], {}
        for gv in range(11):
            ids = [s for s in enrolled if any((s, q, gv) in gidx for q in gq)]
            rows = [gidx[(s, q, gv)] for s in ids for q in gq if (s, q, gv) in gidx]
            owner = torch.tensor([ids.index(gcells[r][0]) for r in rows], device=device)
            for pv in range(11):
                if pv == gv:
                    continue
                prow = [pidx[(s, q, pv)] for s in test for q in pq if (s, q, pv) in pidx]
                if not prow:
                    continue
                sc, am = score(pm, kind, MG[rows], VG[rows], owner, len(ids), MP[prow], VP[prow])
                known = np.array([pcells[r][0] in enr for r in prow])
                ck += [ids[a_] == pcells[r][0] for r, a_, k_ in zip(prow, am, known) if k_]
                for rule, v in sc.items():
                    sk.setdefault(rule, []).append(v[known])
                    su.setdefault(rule, []).append(v[~known])
        ck = np.array(ck)
        res = {'rank1_known': float(ck.mean())}
        for rule in sk:
            k_, u_ = np.concatenate(sk[rule]), np.concatenate(su[rule])
            res[f'{rule}/auroc'] = auroc(k_, u_)
            res[f'{rule}/oscr'] = oscr(k_, ck, u_)
            for f in FARS:
                res[f'{rule}/dir@far{int(f * 100)}'] = dir_at_far(k_, ck, u_, f)
        out.append(res)
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2', nargs='+', required=True)
    ap.add_argument('--out', default='results/p4c_openset')
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    seeds = sorted(int(os.path.basename(d)[4:]) for d in glob.glob(os.path.join(a.stage2[0], 'seed*')))
    modes = ['nm', 'bg', 'cl', 'nm_occ']
    per = {}
    for seed in seeds:
        per[seed] = {}
        for k, d in enumerate(a.stage2):
            pm = load_pm(d, seed, a.device)
            tag = os.path.basename(d.rstrip('/'))
            for kind in (['b1', 'n1n2'] if k == 0 else ['n1n2']):
                name = 'b1' if kind == 'b1' else f'{tag}:n1n2'
                per[seed][name] = {m: run(pm, kind, ds, test, a.device, a.repeats, m) for m in modes}
                print(f'seed {seed} {name}: ' + ' | '.join(
                    f"{m} OSCR raw {per[seed][name][m]['raw/oscr']:.3f} tnorm {per[seed][name][m]['tnorm/oscr']:.3f}"
                    for m in modes), flush=True)
        save_json(per, os.path.join(a.out, 'per_seed.json'))
    write_summary(per, seeds, modes, a.out)


def write_summary(per, seeds, modes, out):
    models = list(per[seeds[0]])
    summ = {mo: {m: {k: mean_ci([per[s][mo][m][k] for s in seeds]) for k in per[seeds[0]][mo][m]}
                 for m in modes} for mo in models}
    save_json(summ, os.path.join(out, 'summary.json'))
    rules = ['raw', 'margin', 'tnorm', 'lr']
    cols = ['auroc', 'oscr', 'dir@far1', 'dir@far5', 'dir@far10']
    lines = []
    for m in modes:
        lines += ['', f'**probe: {m}**', '', '| model | score | ' + ' | '.join(cols) + ' |',
                  '|---' * (len(cols) + 2) + '|']
        for mo in models:
            for r in rules:
                if f'{r}/auroc' not in summ[mo][m]:
                    continue
                lines.append(f'| {mo} | {r} | ' + ' | '.join(
                    f"{100 * summ[mo][m][r + '/' + k]['mean']:.2f} ± {100 * summ[mo][m][r + '/' + k]['ci95']:.2f}"
                    for k in cols) + ' |')
    md = '\n'.join(lines)
    open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8').write(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
