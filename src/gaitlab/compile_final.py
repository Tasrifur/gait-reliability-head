"""Final plug-in protocol table (from results/p5_plugin/final_protocol.json).

Each fold: head fitted + open-set rule selected on the calibration fold,
metrics on the evaluation fold. Values = mean over the 2 evaluation folds
(head: also over 5 seeds); +/- = 95% CI over the 10 (fold, seed) head runs.
Paired sign test: fraction of (fold, seed) runs in which the head beats the frozen
model evaluated on the same fold.

Usage: python -m gaitlab.compile_final -> results/final_plugin_protocol.md
"""
import json

import numpy as np

from .utils import mean_ci

C = ['nm', 'bg', 'cl']
MODES = ['nm', 'bg', 'cl', 'nm_occ']
CLOSED = [('FG-CV mean', lambda r: np.mean([r[f'fgcv>{c}'] for c in C])),
          ('FG-CV CL', lambda r: r['fgcv>cl']),
          ('S1 mismatched gallery', lambda r: np.mean([r[f'S1_{g}>{p}'] for g in C for p in C if g != p])),
          ('S3 corrupted enrollment', lambda r: np.mean([r[f'S3_corrupt>{c}'] for c in C])),
          ('S4 occluded probes', lambda r: np.mean([r[f'S4_occprobe>{c}'] for c in C]))]


def main():
    X = json.load(open('results/p5_plugin/final_protocol.json'))
    L = ['# Final plug-in protocol: frozen official backbones + reliability head calibrated on unseen subjects', '',
         'CASIA-B LT test subjects (075-124, never seen by the backbones) split into 2 folds. Per fold: head '
         'fitted and open-set score rule selected on the CALIBRATION fold only; all metrics on the EVALUATION '
         'fold (25-subject gallery). Mean over the 2 evaluation folds; head ± = 95% CI over 10 (fold, seed) runs; '
         '"wins" = runs in which the head beats the frozen model on the same evaluation fold.', '']
    for bb in X:
        folds = list(X[bb])
        rules_fr = [X[bb][f]['frozen']['rule'] for f in folds]
        rules_h = [X[bb][f]['head'][s]['rule'] for f in folds for s in X[bb][f]['head']]
        L += [f'## {bb}', '',
              f'Selected open-set rule: frozen {rules_fr}; head {dict((r, rules_h.count(r)) for r in set(rules_h))}', '',
              '| metric | frozen | + head | Δ | wins |', '|---|---|---|---|---|']

        def row(name, fn):
            fr = {f: fn(X[bb][f]['frozen']['eval'], X[bb][f]['frozen']['rule']) for f in folds}
            hv, wins = [], 0
            for f in folds:
                for s, h in X[bb][f]['head'].items():
                    v = fn(h['eval'], h['rule'])
                    hv.append(v)
                    wins += v > fr[f]
            a = np.mean(list(fr.values()))
            ci = mean_ci(hv)
            L.append(f'| {name} | {a:.2f} | {ci["mean"]:.2f} ± {ci["ci95"]:.2f} | {ci["mean"] - a:+.2f} | {wins}/{len(hv)} |')
        for name, fn in CLOSED:
            row(name, lambda r, rule, fn=fn: fn(r))
        for m, label in [('nm', 'NM'), ('bg', 'BG'), ('cl', 'CL'), ('nm_occ', 'NM occluded')]:
            for met, mlab in [('auroc', 'AUROC'), ('oscr', 'OSCR'), ('dir@far1', 'DIR@1%')]:
                row(f'Open-set {label} {mlab}', lambda r, rule, m=m, met=met: 100 * r[f'os_{m}/{rule}/{met}'])
        L.append('')
    md = '\n'.join(L)
    open('results/final_plugin_protocol.md', 'w', encoding='utf-8').write(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
