"""Compile the Phase 4 ablation ladder from saved JSON results (no hand-typed numbers).

Usage: python -m gaitlab.compile_ladder  -> results/phase4_ladder.md
"""
import json

import numpy as np

from .utils import mean_ci


def J(p):
    return json.load(open(p))


def seeds_mean(vals):
    r = mean_ci(vals)
    return f"{r['mean']:.2f} ± {r['ci95']:.2f}"


def main():
    rows = []   # (label, params, dict col -> str)
    fg = {'L0': ('results/v0_dppnet_lt_cv', None), 'L1': ('results/abl_cv_uniform', None),
          'L2': ('results/p3_bn', None), 'L3': ('results/p3_bn_aux', None),
          'L4': ('results/p4_2stage', 'fgcv_n1_lam1'), 'L5': ('results/p4_2stage', 'fgcv_n1n2_lam1'),
          'L6': ('results/p4_2stage_occ', 'fgcv_n1n2_lam1')}
    labels = {'L0': 'Original DPP-Net (residual attention)', 'L1': 'Uniform weights (from scratch)',
              'L2': '+ BatchNorm', 'L3': '+ per-strip aux loss = **B1**',
              'L4': '+ N1 per-strip uncertainty (two-stage)', 'L5': '+ N2 precision-weighted prototypes',
              'L6': '+ occlusion-aware uncertainty (**model of record**)'}
    params = {'L0': '685k', 'L1': '126k', 'L2': '126k', 'L3': '126k', 'L4': '143k', 'L5': '143k', 'L6': '143k'}
    cols = {}
    for L, (d, key) in fg.items():
        s = J(f'{d}/summary.json')['evals'][key or 'full_gallery_cv']
        cols[L] = {'FG-CV mean': f"{s['mean']['mean']:.2f} ± {s['mean']['ci95']:.2f}",
                   'FG-CV CL': f"{s['cl']['mean']:.2f}"}
    # scenarios
    ens = J('results/p4b_enrollment/per_seed.json')
    smap = {'L3': 'b1', 'L4': 'p4_2stage:n1_lam1', 'L5': 'p4_2stage:n1n2_lam1', 'L6': 'p4_2stage_occ:n1n2_lam1'}
    pre = J('results/p4e_ladder/per_seed.json')
    pmap = {'L0': 'L0_dppnet', 'L1': 'L1_uniform', 'L2': 'L2_bn'}

    def scen(L, prefix):
        if L in smap:
            vals = [np.mean([ens[s][smap[L]][f'{prefix}>{c}'] for c in ['nm', 'bg', 'cl']]) for s in ens]
        else:
            vals = [np.mean([pre[pmap[L]][s][f'{prefix}>{c}'] for c in ['nm', 'bg', 'cl']]) for s in pre[pmap[L]]]
        return seeds_mean(vals)
    for L in cols:
        cols[L]['S3 corrupted enroll'] = scen(L, 'S3_corrupt')
        cols[L]['S4 occluded probes'] = scen(L, 'S4_occprobe')
        if L in smap:
            off = [np.mean([ens[s][smap[L]][f'S1_{g}>{p}'] for g in 'nm bg cl'.split()
                            for p in 'nm bg cl'.split() if g != p]) for s in ens]
            cols[L]['S1 mismatched gallery'] = seeds_mean(off)
        else:
            cols[L]['S1 mismatched gallery'] = '—'
    # open-set (best rule per model: margin for score-only models, lr for n1n2)
    osj = J('results/p4c_openset/per_seed.json')
    omap = {'L3': ('b1', 'margin'), 'L5': ('p4_2stage:n1n2', 'lr'), 'L6': ('p4_2stage_occ:n1n2', 'lr')}
    for L in cols:
        for mode, met, name in [('nm', 'auroc', 'OS AUROC NM'), ('cl', 'auroc', 'OS AUROC CL'),
                                ('nm', 'dir@far1', 'OS DIR@1% NM'), ('nm_occ', 'oscr', 'OS OSCR occluded')]:
            if L in omap:
                mo, rule = omap[L]
                vals = [100 * osj[s][mo][mode][f'{rule}/{met}'] for s in osj]
            elif L in pmap:
                vals = [100 * pre[pmap[L]][s][f'os_{mode}/margin/{met}'] for s in pre[pmap[L]]]
            else:
                cols[L][name] = '—'
                continue
            cols[L][name] = seeds_mean(vals)
    # faithfulness (CL): contrib-vs-anti separation
    fi = J('results/p4d_interpret/per_seed.json')
    fmap = {'L0': 'attention', 'L3': 'b1', 'L6': 'n1n2'}
    for L in cols:
        if L in fmap:
            vals = [fi[s][fmap[L]]['cl']['auc_contrib'] - fi[s][fmap[L]]['cl']['auc_anti'] for s in fi]
            cols[L]['Expl. separation CL'] = seeds_mean(vals)
        else:
            cols[L]['Expl. separation CL'] = '—'
    names = ['FG-CV mean', 'FG-CV CL', 'S1 mismatched gallery', 'S3 corrupted enroll', 'S4 occluded probes',
             'OS AUROC NM', 'OS AUROC CL', 'OS DIR@1% NM', 'OS OSCR occluded', 'Expl. separation CL']
    lines = ['# Phase 4 ablation ladder (CASIA-B LT, cross-view, 5 seeds, mean ± 95% CI)', '',
             '| Row | Model | Params | ' + ' | '.join(names) + ' |', '|---' * (len(names) + 3) + '|']
    for L in cols:
        lines.append(f'| {L} | {labels[L]} | {params[L]} | ' + ' | '.join(cols[L][n] for n in names) + ' |')
    lines += ['', 'FG-CV: full-gallery cross-view rank-1 (standard LT). S1/S3/S4: enrollment scenarios (mean over NM/BG/CL probes). '
              'OS: open-set, 25 enrolled / 25 unknown x 10 splits, best score rule per model (margin for L0-L3, likelihood ratio for L5-L6); '
              'L4 has no separate open-set run. Expl. separation: deletion flip-AUC (most-important-first minus least-important-first), CL probes; higher = more discriminative explanations.']
    md = '\n'.join(lines)
    open('results/phase4_ladder.md', 'w', encoding='utf-8').write(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
