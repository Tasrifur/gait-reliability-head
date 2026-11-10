"""Multi-dataset table from results/md_eval_<dataset>.json (no hand-typed numbers).

Usage: python -m gaitlab.compile_multi -> results/multidataset_table.md
"""
import json
import os

import numpy as np

from .utils import mean_ci

DS = [('casiab', 'CASIA-B'), ('casiae', 'CASIA-E'), ('oumvlp', 'OU-MVLP')]
TYPES = ['hband', 'vband', 'bottom', 'noise']


def conds(r):
    return sorted({k.split('>')[1] for k in r if k.startswith('fgcv>')})


def metrics(r):
    C = conds(r)
    m = {'FG-CV mean': np.mean([r[f'fgcv>{c}'] for c in C])}
    if 'cl' in C:
        m['FG-CV CL'] = r['fgcv>cl']
    mm = [v for k, v in r.items() if k.startswith('mismatch_')]
    if mm:
        m['Mismatched gallery'] = np.mean(mm)
    m['Corrupted enrollment'] = np.mean([r[f'corrupt_enroll>{c}'] for c in C])
    for t in TYPES:
        m[f'Occluded probes: {t}'] = np.mean([r[f'occ_{t}>{c}'] for c in C])
    for mode in C + ['occ_hband']:
        m[f'Open-set {mode} AUROC'] = 100 * r[f'os_{mode}/auroc']
        m[f'Open-set {mode} DIR@1%'] = 100 * r[f'os_{mode}/dir@far1']
    return m


def main():
    L = ['# Multi-dataset results: B1 vs B1 + reliability head (calibrated on held-out ids)', '',
         'Official test protocols, full-gallery cross-view. Head calibrated with horizontal bands only '
         '(vband / bottom / noise are unseen corruption types). Open-set rule chosen on the calibration ids. '
         'Mean ± 95% CI over training seeds; wins = seeds where the head beats B1.', '']
    for key, label in DS:
        p = f'results/md_eval_{key}.json'
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        seeds = [s for s in R if 'b1' in R[s] and 'head' in R[s]]   # skip seeds still running
        if not seeds:
            continue
        mb = [metrics(R[s]['b1']) for s in seeds]
        mh = [metrics(R[s]['head']) for s in seeds]
        rules = [R[s]['head']['openset_rule'] for s in seeds]
        rules_b = [R[s]['b1']['openset_rule'] for s in seeds]
        L += [f'## {label} ({len(seeds)} seeds; open-set rules B1 {rules_b}, head {rules})', '',
              '| metric | B1 | + head | Δ | wins |', '|---|---|---|---|---|']
        for k in mb[0]:
            a, b = mean_ci([x[k] for x in mb]), mean_ci([x[k] for x in mh])
            w = sum(h[k] > x[k] for h, x in zip(mh, mb))
            L.append(f"| {k} | {a['mean']:.2f} ± {a['ci95']:.2f} | {b['mean']:.2f} ± {b['ci95']:.2f} | "
                     f"{b['mean'] - a['mean']:+.2f} | {w}/{len(seeds)} |")
        L.append('')
    md = '\n'.join(L)
    open('results/multidataset_table.md', 'w', encoding='utf-8').write(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
