"""Aggregate per-seed metrics.json into mean ± 95% CI over seeds.

Usage: python -m gaitlab.aggregate results/<name> [results/<name2> ...]
Writes results/<name>/summary.json and summary.md, prints the table.
"""
import glob
import json
import os
import sys

import numpy as np

from .utils import mean_ci, save_json

CONDS = ['nm', 'bg', 'cl']


def aggregate(run_dir):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, 'seed*', 'metrics.json')))]
    if not runs:
        return None
    out = {'run': os.path.basename(run_dir), 'n_seeds': len(runs),
           'seeds': [r['seed'] for r in runs], 'n_params': runs[0]['n_params'],
           'best_episode': [r['best_episode'] for r in runs], 'evals': {}}
    for key in runs[0]['results']:
        row = {}
        for c in CONDS:
            row[c] = mean_ci([r['results'][key][c]['mean'] for r in runs])
        row['mean'] = mean_ci([np.mean([r['results'][key][c]['mean'] for c in CONDS]) for r in runs])
        out['evals'][key] = row
    lines = [f"### {out['run']}  ({out['n_seeds']} seeds; ± = 95% CI over seeds)", '',
             '| eval | NM | BG | CL | mean |', '|---|---|---|---|---|']
    for key, row in out['evals'].items():
        cells = [f"{row[c]['mean']:.2f} ± {row[c]['ci95']:.2f}" for c in CONDS + ['mean']]
        lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    md = '\n'.join(lines)
    save_json(out, os.path.join(run_dir, 'summary.json'))
    open(os.path.join(run_dir, 'summary.md'), 'w', encoding='utf-8').write(md + '\n')
    return md


if __name__ == '__main__':
    for d in sys.argv[1:]:
        md = aggregate(d)
        print(md if md else f'{d}: no runs found', '\n')
