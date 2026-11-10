"""Phase 4.6: scenario + open-set numbers for the pre-B1 ladder rows, using
exactly the protocols of eval_enrollment (S3/S4) and eval_openset (same splits,
same occlusions, same metrics; scores: raw and margin).

Rows: L0 original DPP-Net (v0 residual attention, cross-view trained)
      L1 uniform  (abl_cv_uniform)
      L2 +BN      (p3_bn)
Usage: python -m gaitlab.ladder_eval --out results/p4e_ladder
"""
import argparse
import os

import numpy as np
import torch

from .data.casiab import CasiaB, LT_GALLERY, LT_PROBES
from .eval_enrollment import fg_rank1, occlude
from .eval_openset import FARS, auroc, dir_at_far, oscr
from .models.dppnet import DPPNet
from .models.protonet import ProtoNet
from .models.heads import HeadNet
from .utils import mean_ci, save_json

ROWS = {'L0_dppnet': ('results/v0_dppnet_lt_cv', lambda: DPPNet(attention='residual')),
        'L1_uniform': ('results/abl_cv_uniform', lambda: DPPNet(attention='uniform')),
        'L2_bn': ('results/p3_bn', lambda: ProtoNet('v0bn'))}
HEADS = {f'H_{h}': (f'results/p5_{h}', (lambda h=h: HeadNet(h)))
         for h in ['matching', 'relation', 'feat', 'tadam', 'metabase']}


@torch.no_grad()
def openset(model, ds, test, device, mode, repeats=10):
    cond = 'nm' if mode == 'nm_occ' else mode
    gq = [ds.qidx(q) for q in LT_GALLERY]
    pq = [ds.qidx(q) for q in LT_PROBES[cond]]
    rng = np.random.RandomState(2024)                       # same splits as eval_openset
    res = []
    for _ in range(repeats):
        enrolled = sorted(rng.choice(test, len(test) // 2, replace=False).tolist())
        enr = set(enrolled)
        sk = {'raw': [], 'margin': []}
        su = {'raw': [], 'margin': []}
        ck = []
        for gv in range(11):
            ids = [s for s in enrolled if any(ds.valid[s, q, gv] for q in gq)]
            sup = [(s, q, gv) for s in ids for q in gq if ds.valid[s, q, gv]]
            X = torch.from_numpy(np.stack([ds.get(*c) for c in sup])).to(device)
            owner = torch.tensor([ids.index(c[0]) for c in sup], device=device)
            for pv in range(11):
                if pv == gv:
                    continue
                qc = [(s, q, pv) for s in test for q in pq if ds.valid[s, q, pv]]
                if not qc:
                    continue
                Q = torch.from_numpy(np.stack([occlude(ds.get(*c), c, 2, 1.0) if mode == 'nm_occ'
                                               else ds.get(*c) for c in qc])).to(device)
                lg = model.logits_unequal(X, owner, len(ids), Q)
                t2 = lg.topk(2, 1)
                known = np.array([c[0] in enr for c in qc])
                scores = {'raw': t2.values[:, 0].cpu().numpy(),
                          'margin': (t2.values[:, 0] - t2.values[:, 1]).cpu().numpy()}
                am = t2.indices[:, 0].cpu().numpy()
                ck += [ids[a] == c[0] for c, a, k in zip(qc, am, known) if k]
                for r in scores:
                    sk[r].append(scores[r][known]); su[r].append(scores[r][~known])
        ck = np.array(ck)
        out = {}
        for r in sk:
            k_, u_ = np.concatenate(sk[r]), np.concatenate(su[r])
            out[f'{r}/auroc'] = auroc(k_, u_)
            out[f'{r}/oscr'] = oscr(k_, ck, u_)
            for f in FARS:
                out[f'{r}/dir@far{int(f * 100)}'] = dir_at_far(k_, ck, u_, f)
        res.append(out)
    return {k: float(np.mean([o[k] for o in res])) for k in res[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/p4e_ladder')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--rows', nargs='+', help='subset of row names (default: L0-L2)')
    a = ap.parse_args()
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    allrows = {**ROWS, **HEADS}
    sel = {r: allrows[r] for r in (a.rows or ROWS)}
    per = {}
    for row, (run_dir, make) in sel.items():
        per[row] = {}
        for seed in range(5):
            m = make().to(a.device).eval()
            m.load_state_dict(torch.load(f'{run_dir}/seed{seed}/best.pt', weights_only=True))
            r = {}
            for p in ['nm', 'bg', 'cl']:
                r[f'S3_corrupt>{p}'] = fg_rank1(m, ds, test, LT_GALLERY, LT_PROBES[p], a.device, g_occ=0.5)
                r[f'S4_occprobe>{p}'] = fg_rank1(m, ds, test, LT_GALLERY, LT_PROBES[p], a.device, p_occ=1.0)
            for mode in ['nm', 'bg', 'cl', 'nm_occ']:
                for k, v in openset(m, ds, test, a.device, mode).items():
                    r[f'os_{mode}/{k}'] = v
            per[row][seed] = r
            print(row, seed, {k: round(v, 3) for k, v in r.items() if k.startswith('S') or 'margin/auroc' in k},
                  flush=True)
            save_json(per, os.path.join(a.out, 'per_seed.json'))
    summ = {row: {k: mean_ci([per[row][s][k] for s in per[row]]) for k in per[row][0]} for row in per}
    save_json(summ, os.path.join(a.out, 'summary.json'))


if __name__ == '__main__':
    main()
