"""Additional result tables and figures, generated from the saved result files.

Tables -> elsarticle/tables/, figures -> elsarticle/figs/.
Usage: python -m gaitlab.latex_tables_extra
"""
import json
import os

import numpy as np

from .latex_tables import C3, J, write

FIG = 'elsarticle/figs'


# ------------------------------------------------------------------ tables
def t_datasets():
    if not os.path.exists('data/cache/casiab_gei.npz'):
        print('t_datasets: dataset caches not built (see README), skipped'); return
    from .calibrate import load_dataset
    from .multi_eval import protocol
    meta = {'casiab': ('CASIA-B', 'NM, BG, CL', 'NM-01--04', 'NM-05/06, BG, CL'),
            'casiae': ('CASIA-E', 'NM, BG, CL; 3 scenes', 'scene 1 NM, H+L', 'scenes 2, 3'),
            'oumvlp': ('OU-MVLP', 'NM', 'seq.\\ 01', 'seq.\\ 00')}
    col = {}
    for k in ['casiab', 'casiae', 'oumvlp']:
        ds, sp, _, _, _ = load_dataset(k)
        g, pr, _ = protocol(k, ds)
        allp = sorted({q for v in pr.values() for q in v})
        gq = [ds.qidx(q) for q in g]
        pq = [ds.qidx(q) for q in allp]
        ng = int(ds.valid[np.ix_(sp['test'], gq, range(len(ds.views)))].sum())
        npb = int(ds.valid[np.ix_(sp['test'], pq, range(len(ds.views)))].sum())
        n, cond, gal, prb = meta[k]
        col[k] = [n, f'{len(ds.subjects):,}', f'{len(ds.views)}', cond, f"{len(sp['train']):,}", f"{len(sp['val'])}",
                  f"{len(sp['test']):,}", gal, prb, f'{ng:,}', f'{npb:,}']
    labels = [None, 'Identities', 'Views', 'Conditions', 'Training IDs', 'Calibration IDs', 'Test IDs',
              'Gallery', 'Probes', 'Gallery samples', 'Probe samples']
    c = r'>{\centering\arraybackslash}p{0.21\columnwidth}'
    L = [r'\begin{tabular}{@{}l' + c * 3 + '@{}}', r'\toprule',          # transposed: one column per dataset
         ' & ' + ' & '.join(col[k][0] for k in col) + r' \\', r'\midrule']
    for i, lab in enumerate(labels[1:], 1):
        if i in (4, 7, 9):
            L.append(r'\midrule')
        L.append(f'{lab} & ' + ' & '.join(col[k][i] for k in col) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    write('t_datasets', '\n'.join(L) + '\n')


def t_reproduction():
    R = J('results/p5_frozen/results.json')
    R.update(J('results/p5_frozen/repro.json'))             # GaitPart, GaitGL (frozen_hybrid repro)
    pub = {'gaitset': ('GaitSet', (95.8, 90.0, 75.4)),
           'gaitpart': ('GaitPart', (96.1, 90.7, 78.7)),
           'gaitgl': ('GaitGL', (97.4, 94.5, 83.8)),
           'gaitbase': ('GaitBase', (97.6, 94.0, 77.4))}
    L = [r'\begin{tabular}{lcccccc}', r'\toprule',
         r' & \multicolumn{3}{c}{Published (OpenGait)} & \multicolumn{3}{c}{Our evaluation} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}', r'Model & NM & BG & CL & NM & BG & CL \\', r'\midrule']
    for k in [k for k in pub if k in R]:
        name, p = pub[k]
        L.append(f'{name} & ' + ' & '.join(f'{x:.1f}' for x in p) + ' & ' +
                 ' & '.join(f"{R[k][f'published_nn>{c}']:.2f}" for c in C3) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    write('t_reproduction', '\n'.join(L) + '\n')


def t_complexity():
    A = J('results/paper_analyses.json')
    T = A['timing']
    L = [r'\begin{tabular}{@{}lrcc@{}}', r'\toprule',
         r'Model & Params & Enroll (ms) & Match (ms) \\', r'\midrule']
    for k, lab in [('B1', 'B1 backbone'), ('B1+head', 'B1 + head')]:
        L.append(f"{lab} & {T[k]['params']:,} & {T[k]['enroll_ms_per_identity_K4']:.2f} & "
                 f"{T[k]['match_ms_per_probe_50way']:.2f}" + r' \\')
    L += [r'\midrule', r'Head on GaitPart, GaitGL & +8.4k & \multicolumn{2}{c}{$<$1\% of backbone} \\',
          r'Head on GaitSet, GaitBase & +16.5k & \multicolumn{2}{c}{$<$1\% of backbone} \\',
          r'\bottomrule', r'\end{tabular}']
    write('t_complexity', '\n'.join(L) + '\n')


# ------------------------------------------------------------------ figures
def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix',                     # Times-compatible math
                         'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6,
                         'lines.linewidth': 1.2, 'lines.markersize': 3.5, 'grid.linewidth': 0.4,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.alpha': 0.3, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,          # embed TrueType fonts (no Type 3)
                         'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
                         'savefig.dpi': 600})                            # raster parts (GEIs) at 600 dpi
    return plt


# Paper palette: cool colors for baselines and conditions, warm colors for the proposed methods.
INDIGO, PERIWINKLE, AMBER, ORANGE, FLAME = '#3d348b', '#7678ed', '#f7b801', '#f18701', '#f35b04'
LAVENDER = '#bcbdf6'                                     # tint of PERIWINKLE, only where a 6th category is needed
COL = {'b1': INDIGO, 'head': FLAME, 'frozen': INDIGO}
FULL, HALF = 7.1, 3.45       # two-column elsarticle (5p): text width and column width, inches
VIEWS = ['0', '18', '36', '54', '72', '90', '108', '126', '144', '162', '180']


def _agg(R, key):
    b = np.array([R[f'b1/{s}'][key] if not isinstance(key, tuple) else R[f'b1/{s}'][key[0]][key[1]]
                  for s in range(5) if f'b1/{s}' in R])
    h = np.array([R[f'head/{s}'][key] if not isinstance(key, tuple) else R[f'head/{s}'][key[0]][key[1]]
                  for s in range(5) if f'head/{s}' in R])
    return b, h


def f_per_view():
    plt = _style()
    R = J('results/paper_analyses.json')['per_view']
    fig, ax = plt.subplots(2, 2, figsize=(HALF, 3.2))          # one-column 2 x 2 layout
    ax = ax.ravel()
    for j, (a, (k, t)) in enumerate(zip(ax, [('nm', 'NM probes'), ('bg', 'BG probes'), ('cl', 'CL probes'),
                                              ('nm_occ', 'Occluded NM probes')])):
        b = np.array([R[f'b1/{s}'][k] for s in range(5)]); h = np.array([R[f'head/{s}'][k] for s in range(5)])
        x = np.arange(11)
        for arr, lab, c, m in [(b, 'B1', COL['b1'], 's'), (h, 'B1 + head', COL['head'], 'o')]:
            a.plot(x, arr.mean(0), marker=m, ms=2.2, color=c, label=lab)
            a.fill_between(x, arr.min(0), arr.max(0), color=c, alpha=0.15, lw=0)
        a.set_xticks(x[::2]); a.set_xticklabels(VIEWS[::2]); a.set_title(t)
        if j >= 2:
            a.set_xlabel('probe view (deg)')
        if j % 2 == 0:
            a.set_ylabel('rank-1 (%)')
    ax[0].legend(loc='lower center')
    plt.tight_layout(); plt.savefig(f'{FIG}/per_view.pdf'); plt.close()


def f_severity():
    plt = _style()
    A = J('results/paper_analyses.json')
    fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.2))
    for a, (sec, xs, xl, t) in zip(ax[:2], [('enroll_sweep', ['0.0', '0.25', '0.5', '0.75', '1.0'],
                                             'fraction of occluded enrollment samples', '(a) Corrupted enrollment'),
                                            ('probe_sweep', ['0', '8', '16', '24', '32'],
                                             'occluder height on probe (rows of 64)', '(b) Occluded probes')]):
        R = A[sec]
        for kind, lab, m in [('b1', 'B1', 's'), ('head', 'B1 + head', 'o')]:
            arr = np.array([[R[f'{kind}/{s}'][x] for x in xs] for s in range(5)])
            xv = [float(x) for x in xs]
            a.errorbar(xv, arr.mean(0), yerr=arr.std(0), marker=m, ms=3, capsize=1.5, color=COL[kind], label=lab)
        a.set_xlabel(xl); a.set_title(t)
    ax[0].set_ylabel('rank-1 (%)'); ax[0].legend()
    R = A['kshot']; a = ax[2]
    for kind, lab, c in [('b1', 'B1', COL['b1']), ('head', 'B1 + head', COL['head'])]:
        for g, ls, m in [('clean', '-', 'o'), ('corrupt', '--', '^')]:
            arr = np.array([[R[f'{kind}/{s}'][f'{g}/K{k}'] for k in [1, 2, 3, 4]] for s in range(5)])
            a.plot([1, 2, 3, 4], arr.mean(0), ls=ls, marker=m, ms=3, color=c, label=f'{lab}, {g}')
    a.set_xticks([1, 2, 3, 4]); a.set_xlabel('enrollment samples $K$'); a.set_title('(c) Enrollment size'); a.legend()
    plt.tight_layout(); plt.savefig(f'{FIG}/severity.pdf'); plt.close()


def f_dirfar():
    plt = _style()
    D = J('results/paper_analyses.json')['dirfar']
    fig, ax = plt.subplots(3, 1, figsize=(HALF, 3.3), sharex=True)        # one column, panels stacked
    for a, (m, t) in zip(ax, [('nm', 'NM probes'), ('cl', 'CL probes'), ('nm_occ', 'Occluded NM probes')]):
        for kind, lab in [('b1', 'B1'), ('head', 'B1 + head')]:
            fa, dr = D[kind][m]
            a.plot(np.array(fa) * 100, np.array(dr) * 100, color=COL[kind], label=lab, lw=1.4)
        a.set_xscale('log'); a.set_ylabel('DIR (%)')
        a.text(0.02, 0.93, t, transform=a.transAxes, va='top', fontsize=7.5)
    ax[-1].set_xlabel('false alarm rate (%)'); ax[0].legend(loc='lower right')
    plt.tight_layout(); plt.savefig(f'{FIG}/dirfar.pdf'); plt.close()


def f_reliability():
    plt = _style()
    R = J('results/paper_analyses.json')['reliability']
    fig, a = plt.subplots(1, 1, figsize=(HALF, 2.7))
    allv = []
    for s, r in R.items():
        a.plot(r['pred'], r['obs'], marker='o', ms=2.5, color=COL['head'], alpha=0.6, lw=0.9)
        allv += r['pred'] + r['obs']
    lo, hi = min(allv), max(allv)
    a.plot([lo, hi], [lo, hi], 'k:', lw=1, label='ideal')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel(r'predicted variance $s_i(q)+s_i(c)$'); a.set_ylabel(r'observed $\|\mu_i(q)-c_i\|^2/D$')
    rho = np.mean([r['spearman'] for r in R.values()])
    a.set_title(f'Spearman $\\rho$ = {rho:.2f}'); a.legend(loc='upper left')
    plt.tight_layout(); plt.savefig(f'{FIG}/reliability.pdf'); plt.close()


def f_robustness_bars():
    """Gain of the head per dataset and scenario (grouped bars)."""
    plt = _style()
    from .compile_multi import metrics
    keys = [('Corrupted enrollment', 'Corrupted\nenrollment'), ('Occluded probes: hband', 'Horizontal\nband$^\\dagger$'),
            ('Occluded probes: bottom', 'Lower\nbody'), ('Open-set occ_hband AUROC', 'Open-set\noccl. AUROC'),
            ('FG-CV mean', 'Clean\ncross-view')]
    fig, a = plt.subplots(1, 1, figsize=(HALF, 2.5))
    w = 0.26
    for j, (ds, lab, c) in enumerate([('casiab', 'CASIA-B', INDIGO), ('casiae', 'CASIA-E', AMBER), ('oumvlp', 'OU-MVLP', FLAME)]):
        R = J(f'results/md_eval_{ds}.json')
        seeds = [s for s in R if 'b1' in R[s] and 'head' in R[s]]
        d = [[metrics(R[s]['head'])[k] - metrics(R[s]['b1'])[k] for s in seeds] for k, _ in keys]
        a.bar(np.arange(len(keys)) + (j - 1) * w, [np.mean(x) for x in d], w, yerr=[np.std(x) for x in d],
              capsize=1.2, color=c, label=lab, error_kw={'elinewidth': 0.6})
    a.axhline(0, color='k', lw=0.6); a.set_xticks(np.arange(len(keys))); a.set_xticklabels([k[1] for k in keys], fontsize=5.8)
    a.set_ylabel('gain over B1 (points)'); a.legend(ncol=3, loc='upper right')
    plt.tight_layout(); plt.savefig(f'{FIG}/robustness_bars.pdf'); plt.close()


def f_heads_bars():
    plt = _style()
    rows = [('Proto', 'results/p3_bn'), ('Matching', 'results/p5_matching'), ('Relation', 'results/p5_relation'),
            ('Meta-B.', 'results/p5_metabase'), ('FEAT', 'results/p5_feat'), ('TADAM', 'results/p5_tadam'),
            ('B1', 'results/p3_bn_aux')]
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.1))
    names = [r[0] for r in rows] + ['Ours']
    m = [J(f'{d}/summary.json')['evals']['full_gallery_cv'] for _, d in rows]
    ours = J('results/p4_2stage_occ/summary.json')['evals']['fgcv_n1n2_lam1']
    for a, key, t in [(ax[0], 'mean', 'Cross-view rank-1 (mean)'), (ax[1], 'cl', 'Cross-view rank-1 (CL)')]:
        vals = [x[key]['mean'] for x in m] + [ours[key]['mean']]
        errs = [x[key]['ci95'] for x in m] + [ours[key]['ci95']]
        cols = [LAVENDER] * (len(rows) - 1) + [INDIGO, FLAME]
        a.bar(names, vals, yerr=errs, capsize=1.5, color=cols, error_kw={'elinewidth': 0.6}); a.set_title(t)
        a.set_ylim(min(vals) - 8, max(vals) + 4)
        a.tick_params(axis='x', labelrotation=30)
        for lab in a.get_xticklabels():
            lab.set_ha('right')
    ax[0].set_ylabel('rank-1 (%)')
    plt.tight_layout(); plt.savefig(f'{FIG}/heads_bars.pdf'); plt.close()


def f_attention_bars():
    plt = _style()
    rows = [('Uniform', 'abl_cv_uniform'), ('Global', 'abl_cv_global'), ('Query', 'abl_cv_query'),
            ('[q; c̄]', 'abl_cv_concat'), ('Residual', 'v0_dppnet_lt_cv')]
    fig, a = plt.subplots(1, 1, figsize=(HALF, 2.2))
    x = np.arange(len(rows)); w = 0.26
    for j, (c, col) in enumerate([('nm', INDIGO), ('bg', PERIWINKLE), ('cl', AMBER)]):
        v = [J(f'results/{r}/summary.json')['evals']['full_gallery_cv'][c] for _, r in rows]
        a.bar(x + (j - 1) * w, [z['mean'] for z in v], w, yerr=[z['ci95'] for z in v], capsize=1.2, color=col,
              label=c.upper(), error_kw={'elinewidth': 0.6})
    a.set_xticks(x); a.set_xticklabels([r[0] for r in rows]); a.set_ylabel('cross-view rank-1 (%)')
    a.set_ylim(0, 84); a.legend(ncol=3, loc='upper center')
    plt.tight_layout(); plt.savefig(f'{FIG}/attention_bars.pdf'); plt.close()


def f_frozen_types():
    plt = _style()
    from .latex_tables import FROZEN_BBS, frozen_occ_types
    X = frozen_occ_types()
    types = [('clean', 'Clean'), ('hband', 'H. band$^\\dagger$'), ('bottom', 'Lower body'), ('vband', 'V. band'), ('noise', 'Noise')]
    bbs = [(bb, f'{n} (frozen)') for bb, n, _ in FROZEN_BBS if bb in X]
    fig, ax = plt.subplots(1, len(bbs), figsize=(FULL, 2.2), sharey=True)
    for a, (bb, t) in zip(ax, bbs):
        fr = [np.mean([d['frozen'][k] for d in X[bb].values()]) for k, _ in types]
        hd = [np.mean([r[k] for d in X[bb].values() for r in d['head'].values()]) for k, _ in types]
        x = np.arange(len(types))
        a.bar(x - 0.2, fr, 0.4, color=COL['frozen'], label='frozen'); a.bar(x + 0.2, hd, 0.4, color=COL['head'], label='+ head')
        a.set_xticks(x); a.set_xticklabels([t_[1] for t_ in types], rotation=35, ha='right'); a.set_title(t)
    ax[0].set_ylabel('rank-1 (%)')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(f'{FIG}/frozen_types.pdf'); plt.close()


def f_baselines():
    """Robustness baselines vs the head and the hybrid, per dataset (results/robust_baselines_*.json)."""
    plt = _style()
    kinds = [('b1', 'Mean (B1)', INDIGO), ('median', 'Median', PERIWINKLE), ('energy', 'Energy mask', LAVENDER),
             ('head_global', 'Pooled var.', AMBER), ('head', 'Head (ours)', ORANGE),
             ('hybrid', 'Head + energy (ours)', FLAME)]
    scen = [('corrupt_enroll', 'Corrupted\nenrollment'), ('occ_hband', 'Horizontal\nband$^\\dagger$'),
            ('occ_bottom', 'Lower\nbody')]
    fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.3))
    w = 0.13
    for a, (ds, t) in zip(ax, [('casiab', 'CASIA-B'), ('casiae', 'CASIA-E'), ('oumvlp', 'OU-MVLP')]):
        R = J(f'results/robust_baselines_{ds}.json')
        seeds = sorted(R)
        sc = [s for s in scen if any(c.split('>')[0] == s[0] for c in R[seeds[0]]['acc'])]
        for j, (k, lab, col) in enumerate(kinds):
            v = [[np.mean([x[k] for c, x in R[s]['acc'].items() if c.split('>')[0] == s_]) for s in seeds]
                 for s_, _ in sc]
            a.bar(np.arange(len(sc)) + (j - 2.5) * w, [np.mean(x) for x in v], w, yerr=[np.std(x) for x in v],
                  capsize=1, color=col, label=lab, error_kw={'elinewidth': 0.5})
        a.set_xticks(np.arange(len(sc))); a.set_xticklabels([s[1] for s in sc]); a.set_title(t)
    ax[0].set_ylabel('rank-1 (%)')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.09, 1, 1)); plt.savefig(f'{FIG}/baselines.pdf'); plt.close()


PARTS = ['head', 'shoulders', 'upper torso', 'lower torso', 'pelvis', 'thighs', 'calves', 'feet']


def _show_gei(a, img, up=8):
    """GEI upsampled by pixel replication, so the embedded image stays sharp at any zoom in the PDF."""
    a.imshow(np.kron(np.asarray(img, dtype=float), np.ones((up, up))), cmap='gray', vmin=0, vmax=float(np.max(img)) or 1,
             interpolation='nearest', extent=(0, img.shape[1], img.shape[0], 0))


def f_corruption_types():
    """One CASIA-B test GEI under the four probe corruptions (occlusions.corrupt, as used in evaluation)."""
    if not os.path.exists('data/cache/casiab_gei.npz'):
        print('f_corruption_types: dataset caches not built (see README), skipped'); return
    plt = _style()
    from .data.casiab import CasiaB, SEQS, VIEWS as CV
    from .occlusions import corrupt
    ds = CasiaB('data/cache/casiab_gei.npz')
    s, q, v = ds.lt_split(12)['test'][3], SEQS.index('nm-05'), CV.index('090')
    x = ds.get(s, q, v)
    kinds = [(None, 'Clean'), ('hband', 'Horizontal band$^\\dagger$'), ('bottom', 'Lower body'),
             ('vband', 'Vertical band'), ('noise', 'Segmentation noise')]
    fig, ax = plt.subplots(1, 5, figsize=(FULL, 1.75))
    for a, (k, t) in zip(ax, kinds):
        _show_gei(a, np.clip((x if k is None else corrupt(x, (s, q, v), k))[0], 0, 1))
        a.set_title(t); a.axis('off')
    plt.tight_layout(w_pad=0.4); plt.savefig(f'{FIG}/corruption_types.pdf'); plt.close()


def f_overview_thumbs():
    """Silhouette thumbnails for the overview diagram (Fig. 1): three enrollment walks of one test identity at
    90 deg, the second band-occluded, and a probe of the same identity at 72 deg; strips of the 8 parts marked."""
    if not os.path.exists('data/cache/casiab_gei.npz'):
        print('f_overview_thumbs: dataset caches not built (see README), skipped'); return
    plt = _style()
    from .data.casiab import CasiaB, SEQS, VIEWS as CV
    ds = CasiaB('data/cache/casiab_gei.npz')
    s = ds.lt_split(12)['test'][3]
    cells = {'ov_enroll1': ('nm-01', '090', None), 'ov_enroll2': ('nm-02', '090', 'hband'),
             'ov_enroll3': ('nm-03', '090', None), 'ov_probe': ('nm-05', '072', None)}
    for name, (q, v, k) in cells.items():
        cell = (s, SEQS.index(q), CV.index(v))
        x = ds.get(*cell)[0]
        if k:
            t, h = 34, 12                                     # a visible band across the thighs
            x = x.copy(); x[t:t + h, :] = 0
        fig = plt.figure(figsize=(0.62, 0.62))
        a = fig.add_axes((0, 0, 1, 1))
        _show_gei(a, np.clip(x[:, 6:58], 0, 1))
        for r in range(1, 8):
            a.axhline(8 * r, color=AMBER, lw=0.35, alpha=0.9)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color('#555555'); sp.set_linewidth(0.5)
        fig.savefig(f'{FIG}/{name}.pdf', bbox_inches=None, pad_inches=0); plt.close(fig)


def f_uncertainty_examples(run='results/p4_2stage_occ'):
    """Predicted per-part variance for one test subject (NM/BG/CL at 90 deg, seed 0) and the median CL/NM
    variance ratio per part over the test set (uncertainty_validation.json of the same run)."""
    if not os.path.exists('data/cache/casiab_gei.npz'):
        print('f_uncertainty_examples: dataset caches not built (see README), skipped'); return
    plt = _style()
    import torch
    from .data.casiab import CasiaB, SEQS, VIEWS as CV
    from . import validate_uncertainty as vu
    vu.RUN = run                                           # load() reads the head of this run
    load = vu.load
    ds = CasiaB('data/cache/casiab_gei.npz')
    s, v = ds.lt_split(12)['test'][3], CV.index('090')
    m = load(0, 'cpu')
    summ = J(f'{run}/uncertainty_validation.json')['summary']
    # one row: [GEI | variance bars] for NM, BG, CL, then the test-set CL/NM ratio. Each GEI sits directly left of
    # its bars, so strip i of the silhouette is aligned with the bar of part i; part names label the first GEI.
    W, H = FULL, 1.85
    bot, top, left = 0.40, 0.24, 0.66                        # inches: x labels, titles, part names
    h = H - bot - top
    seqs = [0, 6, 8]
    cols = np.where(np.max([ds.gei[s, q, v] for q in seqs], axis=(0, 1)) > 0.05 * ds.gei[s, 0, v].max())[0]
    c0, c1 = max(cols.min() - 3, 0), min(cols.max() + 4, ds.gei.shape[-1])   # crop empty columns, same for all
    img0 = ds.gei[s, 0, v][:, c0:c1]
    g = h * img0.shape[1] / img0.shape[0]                    # GEI width that keeps its pixels square
    gap_in, gap_out = 0.04, 0.30
    bw = (W - left - 0.08 - 3 * (g + gap_in + gap_out)) / 4  # width of each bar panel
    fig = plt.figure(figsize=(W, H))
    ax = lambda x, w: fig.add_axes((x / W, bot / H, w / W, h / H))
    x = left
    for k, q in enumerate(seqs):
        with torch.no_grad():
            _, var = m.embed(torch.from_numpy(ds.get(s, q, v))[None])
        img = ds.gei[s, q, v][:, c0:c1]
        a = ax(x, g)
        _show_gei(a, img)
        strip = img.shape[0] / 8
        for r in range(1, 8):
            a.axhline(strip * r, color=AMBER, lw=0.4)
        a.set_xticks([]); a.set_yticks([strip * (i + 0.5) for i in range(8)])
        a.set_yticklabels(PARTS if k == 0 else [])
        a.tick_params(axis='y', length=0)
        for sp in a.spines.values():
            sp.set_visible(False)
        b = ax(x + g + gap_in, bw)
        b.barh(range(8), var[0].numpy(), color=FLAME, height=0.62)
        b.set_ylim(7.5, -0.5); b.set_yticks([]); b.grid(axis='y', visible=False)
        b.set_xlabel('variance $s_i$')
        b.xaxis.set_major_locator(plt.MaxNLocator(3))
        fig.text((x + (g + gap_in + bw) / 2) / W, 1 - 0.06 / H, f'{SEQS[q].upper()}, 90$^\\circ$', ha='center', va='top')
        x += g + gap_in + bw + gap_out
    b = ax(x, bw)
    r = np.array([summ['ratio_cl_nm'][p]['mean'] for p in PARTS])
    b.barh(range(8), r, color=INDIGO, height=0.62); b.axvline(1, color='k', lw=0.6)
    b.set_ylim(7.5, -0.5); b.set_yticks([]); b.grid(axis='y', visible=False)
    b.set_xlabel('CL / NM ratio'); b.set_xticks([0, 0.5, 1, 1.5])
    fig.text((x + bw / 2) / W, 1 - 0.06 / H, 'test set, 5 seeds', ha='center', va='top')
    plt.savefig(f'{FIG}/uncertainty_examples.pdf', bbox_inches=None); plt.close()


def f_deletion_curves():
    """Input-deletion flip curves for coat-wearing probes (results/p4d_interpret/summary.json, 5 seeds)."""
    plt = _style()
    S = J('results/p4d_interpret/summary.json')
    orders = [('contrib', 'margin decomposition (most important first)', FLAME, '-', 'o'),
              ('weight', 'weights (attention / inverse variance)', PERIWINKLE, '--', 's'),
              ('random', 'random order', INDIGO, ':', None),
              ('anti', 'least important first', AMBER, '-.', '^')]
    models = [('attention', 'Attention baseline (L0)'), ('b1', 'B1 (uniform)'), ('n1n2', 'B1 + reliability head')]
    fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.2), sharey=True)
    for a, (mk, t) in zip(ax, models):
        C = S[mk]['cl']['flip_curves']
        for e, lab, c, ls, mrk in orders:
            if e == 'weight' and mk == 'b1':
                continue                                   # B1 weights all parts equally: no weight order (Table faith: --)
            a.plot(range(5), [0] + C[e], color=c, ls=ls, marker=mrk, ms=2.5, label=lab)
        a.set_title(t); a.set_xlabel('parts blanked in the probe ($k$)'); a.set_xticks(range(5))
    ax[0].set_ylabel('fraction of decisions flipped')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.1, 1, 1)); plt.savefig(f'{FIG}/deletion_curves.pdf'); plt.close()



if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    for f in [t_datasets, t_reproduction, f_robustness_bars, f_heads_bars, f_attention_bars, f_frozen_types,
              f_baselines, f_corruption_types, f_overview_thumbs, f_uncertainty_examples, f_deletion_curves]:
        f(); print('ok', f.__name__)
    A = J('results/paper_analyses.json')
    if 'timing' in A:
        for f in [t_complexity, f_per_view, f_severity, f_dirfar, f_reliability]:
            f(); print('ok', f.__name__)
    else:
        print('paper_analyses still running: per-view / severity / dirfar / reliability / complexity pending')
