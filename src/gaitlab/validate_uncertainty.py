"""Phase 4.5: does the learned per-strip variance mean what we claim?

For each two-stage seed (frozen B1 encoder + trained variance head), on TEST subjects:
  1. condition response: median variance per strip for NM/BG/CL; CL/NM, BG/NM ratios
  2. error tracking: Spearman correlation between predicted s_q,i and the actual
     squared discrepancy of the query strip to its own identity's prototype
     (cross-view, gallery NM-01..04 at every other view)
  3. synthetic occlusion: blank one strip (zero its 8 image rows) and measure the
     variance change of that strip vs. the other strips
Writes results/p4_2stage/uncertainty_validation.json and a figure.

Usage: python -m gaitlab.validate_uncertainty
"""
import glob
import json
import os

import numpy as np
import scipy.stats as st
import torch

from .data.casiab import CasiaB, LT_GALLERY, SEQS, VIEWS
from .models.probnet import ProbProtoNet
from .utils import mean_ci, save_json

RUN = 'results/p4_2stage'
PARTS = ['head', 'shoulders', 'upper torso', 'lower torso', 'pelvis', 'thighs', 'calves', 'feet']


@torch.no_grad()
def load(seed, device):
    m = ProbProtoNet('v0bn').to(device).eval()
    b1 = torch.load(f'results/p3_bn_aux/seed{seed}/best.pt', map_location=device, weights_only=True)
    m.cnn.load_state_dict({k[4:]: v for k, v in b1.items() if k.startswith('cnn.')})
    m.load_state_dict(torch.load(f'{RUN}/seed{seed}/var_head.pt', map_location=device, weights_only=True),
                      strict=False)
    return m


@torch.no_grad()
def analyse(m, ds, test, device):
    cells = [(s, q, v) for s in test for q in range(10) for v in range(11) if ds.valid[s, q, v]]
    X = torch.from_numpy(np.stack([ds.get(*c) for c in cells])).to(device)
    MU, VAR = [], []
    for i in range(0, len(X), 512):
        mu, var = m.embed(X[i:i + 512])
        MU.append(mu.cpu()); VAR.append(var.cpu())
    MU, VAR = torch.cat(MU), torch.cat(VAR).numpy()
    idx = {c: i for i, c in enumerate(cells)}
    cond = np.array(['nm' if c[1] < 6 else 'bg' if c[1] < 8 else 'cl' for c in cells])
    out = {'median_var': {k: [float(np.median(VAR[cond == k, i])) for i in range(8)]
                          for k in ['nm', 'bg', 'cl']}}
    out['ratio_cl_nm'] = [out['median_var']['cl'][i] / out['median_var']['nm'][i] for i in range(8)]
    out['ratio_bg_nm'] = [out['median_var']['bg'][i] / out['median_var']['nm'][i] for i in range(8)]

    # 2. error tracking (probe cells only: NM-05/06, BG, CL)
    gq = [ds.qidx(q) for q in LT_GALLERY]
    pv_, err = [], []
    for (s, q, v), i in idx.items():
        if q in gq:
            continue
        d = []
        for gv in range(11):
            if gv == v:
                continue
            g = [MU[idx[(s, k, gv)]] for k in gq if (s, k, gv) in idx]
            if g:
                d.append(((MU[i] - torch.stack(g).mean(0)) ** 2).sum(-1))
        if d:
            pv_.append(VAR[i]); err.append(torch.stack(d).mean(0).numpy())
    pv_, err = np.array(pv_), np.array(err)
    out['spearman_var_vs_error'] = [float(st.spearmanr(pv_[:, i], err[:, i])[0]) for i in range(8)]

    # 3. synthetic occlusion: blank strip j in the image (rows 8j..8j+7)
    sel = np.random.RandomState(0).choice(len(cells), 600, replace=False)
    Xs = X[sel]
    _, v0 = m.embed(Xs)
    occ = np.zeros((8, 8))
    for j in range(8):
        Xo = Xs.clone()
        Xo[:, :, 8 * j:8 * j + 8, :] = 0
        _, vo = m.embed(Xo)
        occ[j] = torch.median(vo / v0, 0).values.cpu().numpy()   # [blanked strip j, strip i]
    out['occlusion_ratio'] = occ.tolist()
    out['occlusion_self_ratio'] = [float(occ[j, j]) for j in range(8)]
    out['occlusion_other_ratio'] = [float(np.median(np.delete(occ[j], j))) for j in range(8)]
    return out


def main():
    import sys
    global RUN
    RUN = sys.argv[1] if len(sys.argv) > 1 else RUN
    device = 'cuda'
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    seeds = sorted(int(os.path.basename(d)[4:]) for d in glob.glob(f'{RUN}/seed*'))
    per = []
    for s in seeds:
        print('seed', s, flush=True)
        per.append(analyse(load(s, device), ds, test, device))
    summ = {}
    for key in ['ratio_cl_nm', 'ratio_bg_nm', 'spearman_var_vs_error',
                'occlusion_self_ratio', 'occlusion_other_ratio']:
        summ[key] = {PARTS[i]: mean_ci([p[key][i] for p in per]) for i in range(8)}
    save_json({'per_seed': per, 'summary': summ}, f'{RUN}/uncertainty_validation.json')
    for key in summ:
        print(f'{key:24s}', ' '.join(f"{p[:6]}={summ[key][p]['mean']:.2f}" for p in PARTS))

    # figure: one subject at 090, NM/BG/CL GEI + variance bars (seed 0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = load(seeds[0], device)
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
    s, v = test[3], VIEWS.index('090')
    with torch.no_grad():
        for k, q in enumerate([0, 6, 8]):
            _, var = m.embed(torch.from_numpy(ds.get(s, q, v))[None].to(device))
            ax[0, k].imshow(ds.gei[s, q, v], cmap='gray')
            ax[0, k].set_title(f'subject {ds.subjects[s]} {SEQS[q]} @090', fontsize=9)
            for r in range(1, 8):
                ax[0, k].axhline(8 * r - 0.5, color='y', lw=.4)
            ax[0, k].axis('off')
            ax[1, k].barh(range(8), var[0].cpu().numpy(), color='tab:red')
            ax[1, k].set_yticks(range(8)); ax[1, k].set_yticklabels(PARTS, fontsize=8)
            ax[1, k].invert_yaxis(); ax[1, k].set_xlabel('predicted variance', fontsize=8)
    ax[0, 3].axis('off')
    r = np.array([summ['ratio_cl_nm'][p]['mean'] for p in PARTS])
    ax[1, 3].barh(range(8), r, color='tab:purple'); ax[1, 3].axvline(1, color='k', lw=.8)
    ax[1, 3].set_yticks(range(8)); ax[1, 3].set_yticklabels(PARTS, fontsize=8); ax[1, 3].invert_yaxis()
    ax[1, 3].set_xlabel('median variance CL / NM (test, 5 seeds)', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{RUN}/uncertainty_examples.png', dpi=90)


if __name__ == '__main__':
    main()
