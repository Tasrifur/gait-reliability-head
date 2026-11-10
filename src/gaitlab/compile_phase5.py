"""Compile Phase 5 tables from saved JSON (no hand-typed numbers).

Usage: python -m gaitlab.compile_phase5 -> results/phase5_tables.md
"""
import json

import numpy as np

from .utils import mean_ci

C = ['nm', 'bg', 'cl']


def J(p):
    return json.load(open(p))


def ms(vals, scale=1.0):
    r = mean_ci([scale * v for v in vals])
    return f"{r['mean']:.2f} ± {r['ci95']:.2f}"


def main():
    L = []
    # ---------------- A. controlled few-shot heads (same encoder, protocol, budget) ----------------
    heads = J('results/p5_heads_eval/per_seed.json')
    heads.update(J('results/p5_heads_eval_metabase/per_seed.json'))
    pre = J('results/p4e_ladder/per_seed.json')
    ens = J('results/p4b_enrollment/per_seed.json')
    osj = J('results/p4c_openset/per_seed.json')
    rows = [('Prototypical (= L2)', 'results/p3_bn', ('pre', 'L2_bn'), '126k'),
            ('Matching (no FCE)', 'results/p5_matching', ('h', 'H_matching'), '126k'),
            ('Relation', 'results/p5_relation', ('h', 'H_relation'), '0.67M'),
            ('Meta-Baseline', 'results/p5_metabase', ('h', 'H_metabase'), '126k'),
            ('FEAT', 'results/p5_feat', ('h', 'H_feat'), '3.3M'),
            ('TADAM-style', 'results/p5_tadam', ('h', 'H_tadam'), '0.91M'),
            ('B1: Proto + per-strip aux (ours)', 'results/p3_bn_aux', ('b1', None), '126k'),
            ('Model of record: B1 + N1/N2/N4 occ-aware (ours)', 'results/p4_2stage_occ', ('ours', None), '143k')]
    cols = ['FG-CV mean', 'FG-CV CL', 'S3 corrupted enroll', 'S4 occluded probes',
            'OS AUROC NM', 'OS AUROC CL', 'OS DIR@1% NM', 'OS OSCR occluded']
    L += ['## A. Controlled comparison of few-shot heads', '',
          'Identical v0+BN encoder, data, episodic budget (≤8,000 episodes, early stopping), optimiser and '
          'cross-view protocol; only the head differs. CASIA-B LT test, 5 seeds, mean ± 95% CI. Open-set: '
          'margin score for all heads (our model: likelihood ratio).', '',
          '| Head | Params | ' + ' | '.join(cols) + ' |', '|---' * (len(cols) + 2) + '|']
    for name, rd, (src, key), par in rows:
        sj = J(f'{rd}/summary.json')
        if src == 'ours':
            fg = sj['evals']['fgcv_n1n2_lam1']
        else:
            fg = sj['evals']['full_gallery_cv']
        c = {'FG-CV mean': f"{fg['mean']['mean']:.2f} ± {fg['mean']['ci95']:.2f}", 'FG-CV CL': f"{fg['cl']['mean']:.2f}"}
        if src in ('pre', 'h'):
            d = pre[key] if src == 'pre' else heads[key]
            sd = list(d)
            c['S3 corrupted enroll'] = ms([np.mean([d[s][f'S3_corrupt>{x}'] for x in C]) for s in sd])
            c['S4 occluded probes'] = ms([np.mean([d[s][f'S4_occprobe>{x}'] for x in C]) for s in sd])
            for m, met, nm in [('nm', 'auroc', 'OS AUROC NM'), ('cl', 'auroc', 'OS AUROC CL'),
                               ('nm', 'dir@far1', 'OS DIR@1% NM'), ('nm_occ', 'oscr', 'OS OSCR occluded')]:
                c[nm] = ms([d[s][f'os_{m}/margin/{met}'] for s in sd], 100)
        else:
            mk = 'b1' if src == 'b1' else 'p4_2stage_occ:n1n2_lam1'
            ok, rule = ('b1', 'margin') if src == 'b1' else ('p4_2stage_occ:n1n2', 'lr')
            c['S3 corrupted enroll'] = ms([np.mean([ens[s][mk][f'S3_corrupt>{x}'] for x in C]) for s in ens])
            c['S4 occluded probes'] = ms([np.mean([ens[s][mk][f'S4_occprobe>{x}'] for x in C]) for s in ens])
            for m, met, nm in [('nm', 'auroc', 'OS AUROC NM'), ('cl', 'auroc', 'OS AUROC CL'),
                               ('nm', 'dir@far1', 'OS DIR@1% NM'), ('nm_occ', 'oscr', 'OS OSCR occluded')]:
                c[nm] = ms([osj[s][ok][m][f'{rule}/{met}'] for s in osj], 100)
        L.append(f'| {name} | {par} | ' + ' | '.join(c[k] for k in cols) + ' |')

    # ---------------- B. frozen supervised backbones + plug-in head ----------------
    P = J('results/p5_plugin/eval_per_seed.json')
    L += ['', '## B. Frozen supervised backbones (OpenGait, CASIA-B LT) and the plug-in reliability head', '',
          'Silhouette sequences; our few-shot protocols (prototype enrollment, no retraining). "+ head" = '
          'N1/N2 occlusion-aware variance head fitted on the 74 training subjects (5 head seeds). Open-set: '
          'raw max-score for every row.', '',
          '| Model | Params | FG-CV mean | FG-CV CL | S1 mismatched | S3 corrupted | S4 occluded | '
          'OS AUROC NM | OS AUROC CL | OS DIR@1% NM | OS DIR@1% BG | OS AUROC occl. | OS DIR@1% occl. |',
          '|---' * 13 + '|']
    npar = {'gaitbase': '7.3M', 'gaitset': '2.6M'}
    for bb in P:
        for kind, label in [('frozen', 'frozen'), ('n1n2', '+ head')]:
            seeds = ['0'] if kind == 'frozen' else list(P[bb])
            rs = [P[bb][s][kind] for s in seeds]
            f = (lambda fn: f'{np.mean([fn(r) for r in rs]):.2f}') if kind == 'frozen' else \
                (lambda fn: ms([fn(r) for r in rs]))
            vals = [f(lambda r: np.mean([r[f'fgcv>{x}'] for x in C])), f(lambda r: r['fgcv>cl']),
                    f(lambda r: np.mean([r[f'S1_{g}>{p}'] for g in C for p in C if g != p])),
                    f(lambda r: np.mean([r[f'S3_corrupt>{x}'] for x in C])),
                    f(lambda r: np.mean([r[f'S4_occprobe>{x}'] for x in C])),
                    f(lambda r: 100 * r['os_nm/raw/auroc']), f(lambda r: 100 * r['os_cl/raw/auroc']),
                    f(lambda r: 100 * r['os_nm/raw/dir@far1']), f(lambda r: 100 * r['os_bg/raw/dir@far1']),
                    f(lambda r: 100 * r['os_nm_occ/raw/auroc']), f(lambda r: 100 * r['os_nm_occ/raw/dir@far1'])]
            L.append(f"| {bb} {label} | {npar[bb]}{'+16k' if kind != 'frozen' else ''} | " + ' | '.join(vals) + ' |')

    # ---------------- C. cross-fit ----------------
    X = J('results/p5_plugin/crossfit.json')
    L += ['', '## C. Cross-fit check: head fitted on 25 UNSEEN test subjects, evaluated on the other 25 '
          '(2 folds x 3 seeds; 25-subject gallery, so absolute values are higher than in B)', '',
          '| Backbone | metric | frozen | + head | Δ |', '|---|---|---|---|---|']
    mets = [('FG-CV mean', lambda r: np.mean([r[f'fgcv>{x}'] for x in C])), ('FG-CV CL', lambda r: r['fgcv>cl']),
            ('S3 corrupted', lambda r: np.mean([r[f'S3_corrupt>{x}'] for x in C])),
            ('S4 occluded', lambda r: np.mean([r[f'S4_occprobe>{x}'] for x in C])),
            ('OS AUROC CL', lambda r: 100 * r['os_cl/raw/auroc']), ('OS DIR@1% NM', lambda r: 100 * r['os_nm/raw/dir@far1']),
            ('OS AUROC occluded', lambda r: 100 * r['os_nm_occ/raw/auroc'])]
    for bb in X:
        for nm, fn in mets:
            a = np.mean([fn(X[bb][f]['frozen']) for f in X[bb]])
            b = np.mean([np.mean([fn(r) for r in X[bb][f]['n1n2'].values()]) for f in X[bb]])
            L.append(f'| {bb} | {nm} | {a:.2f} | {b:.2f} | {b - a:+.2f} |')
    md = '\n'.join(L)
    open('results/phase5_tables.md', 'w', encoding='utf-8').write('# Phase 5 tables\n\n' + md + '\n')
    print(md)


if __name__ == '__main__':
    main()
