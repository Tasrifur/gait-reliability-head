"""Generate the manuscript's LaTeX tables directly from saved result JSON files.

Writes elsarticle/tables/*.tex. Every number in the tables comes from results/.
Usage: python -m gaitlab.latex_tables
"""
import json
import os

import numpy as np

from .compile_multi import metrics as md_metrics
from .utils import mean_ci

OUT = 'elsarticle/tables'
C3 = ['nm', 'bg', 'cl']
FROZEN_BBS = [('gaitbase', 'GaitBase', '7.3M'), ('gaitgl', 'GaitGL', '3.1M'), ('gaitpart', 'GaitPart', '1.2M'),
              ('gaitset', 'GaitSet', '2.6M')]


def J(p):
    return json.load(open(p))


def pm(vals, d=2):
    r = mean_ci(vals)
    return f"{r['mean']:.{d}f}{{\\scriptsize$\\pm${r['ci95']:.{d}f}}}"


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f'{name}.tex'), 'w', encoding='utf-8').write(body)
    print('wrote', name)


def delta(a, b):
    d = np.mean(b) - np.mean(a)
    s = f'{d:+.2f}'
    return f'\\textbf{{{s}}}' if d >= 0.5 else s      # bold only non-trivial gains


# ---------------------------------------------------------------- T1 multi-dataset
def t_multidataset():
    rows = [('Clean, full-gallery cross-view', 'FG-CV mean'),
            ('Clothing change (CL)', 'FG-CV CL'),
            ('Mismatched-condition gallery', 'Mismatched gallery'),
            ('Corrupted enrollment', 'Corrupted enrollment'),
            ('Occluded probes: horizontal band$^\\dagger$', 'Occluded probes: hband'),
            ('Occluded probes: lower body', 'Occluded probes: bottom'),
            ('Occluded probes: vertical band', 'Occluded probes: vband'),
            ('Occluded probes: segmentation noise', 'Occluded probes: noise'),
            ('Open-set AUROC, NM probes', 'Open-set nm AUROC'),
            ('Open-set DIR@1\\%FAR, NM probes', 'Open-set nm DIR@1%'),
            ('Open-set AUROC, occluded probes', 'Open-set occ_hband AUROC'),
            ('Open-set DIR@1\\%FAR, occluded probes', 'Open-set occ_hband DIR@1%')]
    data = {}
    for k in ['casiab', 'casiae', 'oumvlp']:
        R = J(f'results/md_eval_{k}.json')
        seeds = [s for s in R if 'b1' in R[s] and 'head' in R[s]]
        data[k] = ([md_metrics(R[s]['b1']) for s in seeds], [md_metrics(R[s]['head']) for s in seeds])
    L = ['\\begin{tabular}{l' + 'ccc' * 3 + '}', '\\toprule',
         ' & \\multicolumn{3}{c}{CASIA-B (5 seeds)} & \\multicolumn{3}{c}{CASIA-E (3 seeds)} & '
         '\\multicolumn{3}{c}{OU-MVLP (3 seeds)} \\\\',
         '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
         'Metric & B1 & +Head & $\\Delta$ & B1 & +Head & $\\Delta$ & B1 & +Head & $\\Delta$ \\\\', '\\midrule']
    for label, key in rows:
        cells = []
        for k in ['casiab', 'casiae', 'oumvlp']:
            b, h = data[k]
            if key not in b[0]:
                cells += ['--', '--', '--']
                continue
            vb, vh = [x[key] for x in b], [x[key] for x in h]
            cells += [f'{np.mean(vb):.2f}', f'{np.mean(vh):.2f}', delta(vb, vh)]
        L.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
        if label.startswith('Mismatched') or label.startswith('Occluded probes: segm'):
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}']
    write('t_multidataset', '\n'.join(L) + '\n')


# ---------------------------------------------------------------- T2 frozen backbones
def t_frozen():
    X = J('results/p5_plugin/final_protocol.json')
    rows = [('Clean cross-view', lambda r, rule: np.mean([r[f'fgcv>{c}'] for c in C3])),
            ('Clothing change (CL)', lambda r, rule: r['fgcv>cl']),
            ('Mismatched-condition gallery', lambda r, rule: np.mean([r[f'S1_{g}>{p}'] for g in C3 for p in C3 if g != p])),
            ('Corrupted enrollment', lambda r, rule: np.mean([r[f'S3_corrupt>{c}'] for c in C3])),
            ('Occluded probes (horizontal band)', lambda r, rule: np.mean([r[f'S4_occprobe>{c}'] for c in C3])),
            ('Open-set AUROC, CL', lambda r, rule: 100 * r[f'os_cl/{rule}/auroc']),
            ('Open-set OSCR, CL', lambda r, rule: 100 * r[f'os_cl/{rule}/oscr']),
            ('Open-set AUROC, occluded', lambda r, rule: 100 * r[f'os_nm_occ/{rule}/auroc']),
            ('Open-set OSCR, occluded', lambda r, rule: 100 * r[f'os_nm_occ/{rule}/oscr']),
            ('Open-set DIR@1\\%FAR, NM', lambda r, rule: 100 * r[f'os_nm/{rule}/dir@far1'])]
    bbs = [b for b in FROZEN_BBS if b[0] in X]
    L = ['\\begin{tabular}{l' + 'ccc' * len(bbs) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{n} ({p})}}' for _, n, p in bbs) + ' \\\\',
         ''.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(bbs))),
         'Metric' + ' & Frozen & +Head & Wins' * len(bbs) + ' \\\\', '\\midrule']
    for label, fn in rows:
        cells = []
        for bb, _, _ in bbs:
            folds = list(X[bb])
            fr = {f: fn(X[bb][f]['frozen']['eval'], X[bb][f]['frozen']['rule']) for f in folds}
            hv, wins = [], 0
            for f in folds:
                for s, h in X[bb][f]['head'].items():
                    v = fn(h['eval'], h['rule'])
                    hv.append(v); wins += v > fr[f]
            a = np.mean(list(fr.values()))
            b = np.mean(hv)
            cells += [f'{a:.2f}', (f'\\textbf{{{b:.2f}}}' if b - a >= 0.5 else f'{b:.2f}'), f'{wins}/{len(hv)}']
        L.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
        if label.startswith('Occluded'):
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}']
    write('t_frozen', '\n'.join(L) + '\n')


# ---------------------------------------------------------------- T3 ladder (from compile_ladder output)
def md_table_to_tex(md_path, header_match, colspec, drop_cols=(), rename=None):
    lines = open(md_path, encoding='utf-8').read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('|') and header_match in l)
    rows = []
    for l in lines[i:]:
        if not l.startswith('|'):
            break
        cells = [c.strip() for c in l.strip('|').split('|')]
        if set(''.join(cells)) <= set('-: '):
            continue
        rows.append([c for j, c in enumerate(cells) if j not in drop_cols])
    esc = lambda c: (c.replace('**', '').replace('%', '\\%').replace('±', '$\\pm$').replace('_', '\\_')
                     .replace('—', '--').replace('≥', '$\\geq$'))
    if rename:
        rows[0] = rename
    L = [f'\\begin{{tabular}}{{{colspec}}}', '\\toprule', ' & '.join(esc(c) for c in rows[0]) + ' \\\\', '\\midrule']
    L += [' & '.join(esc(c) for c in r) + ' \\\\' for r in rows[1:]]
    L += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(L) + '\n'


RELABEL = [('Original DPP-Net (residual attention)', 'Attention baseline (residual attention)'),
           ('(model of record)', '(full model, ours)'),
           ('Model of record: B1 + N1/N2/N4 occ-aware (ours)', 'B1 + reliability head, occlusion-aware (ours)')]


def relabel(s):
    for a, b in RELABEL:
        s = s.replace(a, b)
    return s


def t_ladder():
    write('t_ladder', relabel(md_table_to_tex(
        'results/phase4_ladder.md', '| Row |', 'llrccccccc', drop_cols=(5, 9, 11),
        rename=['', 'Model', 'Params', 'FG-CV', 'FG-CV CL', 'Corrupted enr.', 'Occluded probes',
                'OS AUROC NM', 'OS DIR@1% NM', 'Expl. sep.'])))


def t_heads():
    write('t_heads', relabel(md_table_to_tex(
        'results/phase5_tables.md', '| Head |', 'lrcccccccc',
        rename=['Head', 'Params', 'FG-CV', 'FG-CV CL', 'Corrupted enr.', 'Occluded probes',
                'OS AUROC NM', 'OS AUROC CL', 'OS DIR@1% NM', 'OS OSCR occl.'])))


def t_attention():
    rows = [('Uniform (from scratch)', 'abl_cv_uniform', None), ('Global learned weights', 'abl_cv_global', None),
            ('Query-only MLP$(q)$', 'abl_cv_query', None), ('MLP$([q;\\bar c])$', 'abl_cv_concat', None),
            ('Residual MLP$([q;q-\\bar c])$ (L0)', 'v0_dppnet_lt_cv', None)]
    L = ['\\begin{tabular}{lrcccc}', '\\toprule', 'Part weighting & Params & NM & BG & CL & Mean \\\\', '\\midrule']
    for label, run, _ in rows:
        S = J(f'results/{run}/summary.json')
        e = S['evals']['full_gallery_cv']
        L.append(f"{label} & {S['n_params']:,} & " + ' & '.join(f"{e[c]['mean']:.2f}" for c in C3) +
                 f" & {e['mean']['mean']:.2f}{{\\scriptsize$\\pm${e['mean']['ci95']:.2f}}} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    write('t_attention', '\n'.join(L) + '\n')


def t_faithfulness():
    S = J('results/p4d_interpret/summary.json')
    names = {'attention': 'Attention baseline (L0)', 'b1': 'B1 (uniform)', 'n1n2': 'B1 + reliability head'}
    L = ['\\begin{tabular}{llccccc}', '\\toprule',
         'Model & Probe & Random & Contrib. & Weights & Least-first & Separation \\\\', '\\midrule']
    for k in ['attention', 'b1', 'n1n2']:
        for c in C3:
            s = S[k][c]
            w = '--' if k == 'b1' else f"{s['gain_weight']['mean']:+.3f}"
            sep = s['gain_contrib']['mean'] - s['gain_anti']['mean']
            L.append(f"{names[k] if c == 'nm' else ''} & {c.upper()} & {s['auc_random']['mean']:.3f} & "
                     f"{s['gain_contrib']['mean']:+.3f} & {w} & {s['gain_anti']['mean']:+.3f} & {sep:.2f} \\\\")
        L.append('\\midrule' if k != 'n1n2' else '')
    L += ['\\bottomrule', '\\end{tabular}']
    write('t_faithfulness', '\n'.join(x for x in L if x) + '\n')


def t_uncertainty():
    S = J('results/p4_2stage_occ/uncertainty_validation.json')['summary']
    parts = ['head', 'shoulders', 'upper torso', 'lower torso', 'pelvis', 'thighs', 'calves', 'feet']
    L = ['\\begin{tabular}{l' + 'c' * 8 + '}', '\\toprule', 'Statistic & ' + ' & '.join(p.title() for p in parts) + ' \\\\',
         '\\midrule']
    for label, key, fmt in [('Variance ratio CL/NM', 'ratio_cl_nm', '{:.2f}'), ('Variance ratio BG/NM', 'ratio_bg_nm', '{:.2f}'),
                            ('Spearman($s$, error)', 'spearman_var_vs_error', '{:.2f}'),
                            ('Blank this strip: own var.\\ $\\times$', 'occlusion_self_ratio', '{:.1f}'),
                            ('\\quad other strips $\\times$', 'occlusion_other_ratio', '{:.2f}')]:
        L.append(label + ' & ' + ' & '.join(fmt.format(S[key][p]['mean']) for p in parts) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    write('t_uncertainty', '\n'.join(L) + '\n')


def frozen_occ_types():
    """Frozen/head rank-1 per probe corruption: the original run (GaitBase, GaitSet) and, for backbones
    added later, the identical evaluation inside frozen_hybrid."""
    X = J('results/p5_plugin/final_occ_types.json')
    H = J('results/p5_plugin/frozen_hybrid.json')
    for bb in H:
        if bb not in X and len(H[bb]) == 2:
            X[bb] = {f: {'frozen': d['frozen'], 'head': d['head']} for f, d in H[bb].items()}
    return X


def t_frozen_types():
    p = 'results/p5_plugin/final_occ_types.json'
    if not os.path.exists(p):
        print('t_frozen_types: results pending'); return
    X = frozen_occ_types()
    types = [('clean', 'Clean'), ('hband', 'Horizontal band$^\\dagger$'), ('bottom', 'Lower body'),
             ('vband', 'Vertical band'), ('noise', 'Segmentation noise')]
    bbs = [b for b in FROZEN_BBS if b[0] in X]
    L = ['\\begin{tabular}{l' + 'ccc' * len(bbs) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{n}}}' for _, n, _ in bbs) + ' \\\\',
         ''.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(bbs))),
         'Probe corruption' + ' & Frozen & +Head & Wins' * len(bbs) + ' \\\\', '\\midrule']
    for t, label in types:
        cells = []
        for bb, _, _ in bbs:
            fr, hv, w = [], [], 0
            for f, d in X[bb].items():
                fr.append(d['frozen'][t])
                for s, r in d['head'].items():
                    hv.append(r[t]); w += r[t] > d['frozen'][t]
            a, b = np.mean(fr), np.mean(hv)
            cells += [f'{a:.2f}', (f'\\textbf{{{b:.2f}}}' if b - a >= 0.5 else f'{b:.2f}'), f'{w}/{len(hv)}']
        L.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    write('t_frozen_types', '\n'.join(L) + '\n')


def t_robust_baselines():
    kinds = ['b1', 'median', 'energy', 'head_global', 'head', 'hybrid']
    scen = [('fgcv', 'Clean'), ('corrupt_enroll', 'Corrupted enrollment'),
            ('occ_hband', 'Horizontal band$^\\dagger$'), ('occ_bottom', 'Lower body'),
            ('occ_vband', 'Vertical band'), ('occ_noise', 'Segmentation noise')]
    L = ['\\begin{tabular}{ll' + 'c' * len(kinds) + '}', '\\toprule',
         ' & & \\multicolumn{4}{c}{Baselines} & \\multicolumn{2}{c}{Ours} \\\\',
         '\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}',
         'Dataset & Scenario & Mean & Median & Energy mask & Pooled var. & Head & Head + energy \\\\']
    for k, name in [('casiab', 'CASIA-B'), ('casiae', 'CASIA-E'), ('oumvlp', 'OU-MVLP')]:
        R = J(f'results/robust_baselines_{k}.json')
        seeds = sorted(R)
        L.append('\\midrule')
        first = True
        for sc, label in scen:
            keys = [c for c in R[seeds[0]]['acc'] if c.split('>')[0] == sc]
            if not keys:
                continue
            v = {m: [np.mean([R[s]['acc'][c][m] for c in keys]) for s in seeds] for m in kinds}
            mean = {m: round(float(np.mean(v[m])), 1) for m in kinds}
            best = max(mean.values())
            bold = best - min(mean.values()) >= 0.5         # no bold where all scorers are tied (near chance)
            cells = [(f'\\textbf{{{mean[m]:.1f}}}' if bold and mean[m] == best else f'{mean[m]:.1f}')
                     + f'{{\\scriptsize$\\pm${np.std(v[m]):.1f}}}' for m in kinds]
            L.append(f"{f'{name} ({len(seeds)})' if first else ''} & {label} & " + ' & '.join(cells) + ' \\\\')
            first = False
    L += ['\\bottomrule', '\\end{tabular}']
    write('t_robust_baselines', '\n'.join(L) + '\n')


def t_frozen_hybrid():
    p = 'results/p5_plugin/frozen_hybrid.json'
    if not os.path.exists(p):
        print('t_frozen_hybrid: results pending'); return
    X = J(p)
    bbs = [(b, n) for b, n, _ in FROZEN_BBS if len(X.get(b, {})) == 2]
    scen = [('clean', 'Clean'), ('corrupt_enroll', 'Corr.\\ enroll.'), ('hband', 'H.\\ band$^\\dagger$'),
            ('bottom', 'Lower body'), ('vband', 'V.\\ band'), ('noise', 'Noise')]
    kinds = [('frozen', 'Frozen'), ('energy', 'Energy mask'), ('head', 'Head'), ('hybrid', 'Head + energy')]
    L = ['\\begin{tabular}{ll' + 'c' * len(scen) + '}', '\\toprule',
         'Backbone & Scorer & ' + ' & '.join(l for _, l in scen) + ' \\\\']
    for bb, name in bbs:
        mean = {}
        for k, _ in kinds:
            for sc, _ in scen:
                v = [x for d in X[bb].values()
                     for x in ([d[k][sc]] if k in ('frozen', 'energy') else [r[sc] for r in d[k].values()])]
                mean[k, sc] = round(float(np.mean(v)), 1)
        L.append('\\midrule')
        for i, (k, label) in enumerate(kinds):
            cells = []
            for sc, _ in scen:
                col = [mean[kk, sc] for kk, _ in kinds]
                bold = max(col) - min(col) >= 0.5 and mean[k, sc] == max(col)
                cells.append(f'\\textbf{{{mean[k, sc]:.1f}}}' if bold else f'{mean[k, sc]:.1f}')
            L.append(f"{name if i == 0 else ''} & {label} & " + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    write('t_frozen_hybrid', '\n'.join(L) + '\n')


def wdist_runs(bb, X=None):
    """Per-run rows of frozen_wdist.json: frozen, likelihood head and weighted distance at the calibrated gamma."""
    X = X or J('results/p5_plugin/frozen_wdist.json')
    out = []
    for f in X[bb].values():
        for r in f['seeds'].values():
            g = str(float(r['wd']['gamma_sel']))
            out.append({'frozen': f['frozen'], 'head': r['head'], 'wd': r['wd']['eval'][g], 'gamma': float(g)})
    return out


def t_frozen_wdist():
    p = 'results/p5_plugin/frozen_wdist.json'
    if not os.path.exists(p):
        print('t_frozen_wdist: results pending'); return
    X = J(p)
    bbs = [(b, n) for b, n, _ in FROZEN_BBS if len(X.get(b, {})) == 2]
    scen = [('clean', 'Clean'), ('cl', 'CL'), ('mismatch', 'Mism.'), ('corrupt_enroll', 'Corr.\\ enroll.'),
            ('hband', 'H.\\ band$^\\dagger$'), ('bottom', 'Lower body'), ('vband', 'V.\\ band'), ('noise', 'Noise')]
    L = ['\\begin{tabular}{ll' + 'c' * len(scen) + '}', '\\toprule',
         'Backbone & Scorer & ' + ' & '.join(l for _, l in scen) + ' \\\\']
    for bb, name in bbs:
        R = wdist_runs(bb, X)
        gam = sorted({r['gamma'] for r in R})
        kinds = [('frozen', 'Frozen'), ('head', 'Likelihood head'),
                 ('wd', 'Weighted distance ($\\hat\\gamma\\in\\{' + ','.join(f'{g:g}' for g in gam) + '\\}$)'
                  if len(gam) > 1 else f'Weighted distance ($\\hat\\gamma={gam[0]:g}$)')]
        mean = {(k, sc): round(float(np.mean([r[k][sc] for r in R])), 1) for k, _ in kinds for sc, _ in scen}
        L.append('\\midrule')
        for i, (k, label) in enumerate(kinds):
            cells = []
            for sc, _ in scen:
                col = [mean[kk, sc] for kk, _ in kinds]
                bold = max(col) - min(col) >= 0.5 and mean[k, sc] == max(col)
                cells.append(f'\\textbf{{{mean[k, sc]:.1f}}}' if bold else f'{mean[k, sc]:.1f}')
            L.append(f"{name if i == 0 else ''} & {label} & " + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    write('t_frozen_wdist', '\n'.join(L) + '\n')


if __name__ == '__main__':
    for f in [t_multidataset, t_frozen, t_ladder, t_heads, t_attention, t_faithfulness, t_uncertainty, t_frozen_types,
              t_robust_baselines, t_frozen_hybrid, t_frozen_wdist]:
        f()
