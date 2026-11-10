"""Additional manuscript tables and figures (v3 polish), generated from saved results.

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
            'casiae': ('CASIA-E', 'NM, BG, CL, 3 scenes', 'scene 1 NM, H+L', 'scenes 2/3'),
            'oumvlp': ('OU-MVLP', 'NM', 'seq. 01', 'seq. 00')}
    rows = []
    for k in ['casiab', 'casiae', 'oumvlp']:
        ds, sp, _, _, _ = load_dataset(k)
        g, pr, _ = protocol(k, ds)
        allp = sorted({q for v in pr.values() for q in v})
        gq = [ds.qidx(q) for q in g]
        pq = [ds.qidx(q) for q in allp]
        ng = int(ds.valid[np.ix_(sp['test'], gq, range(len(ds.views)))].sum())
        npb = int(ds.valid[np.ix_(sp['test'], pq, range(len(ds.views)))].sum())
        n, cond, gal, prb = meta[k]
        rows.append(f"{n} & {len(ds.subjects):,} & {len(ds.views)} & {cond} & {len(sp['train']):,} & "
                    f"{len(sp['val'])} & {len(sp['test']):,} & {gal} & {prb} & {ng:,} & {npb:,} \\\\")
    L = [r'\begin{tabular}{lrrlrrrllrr}', r'\toprule',
         r'Dataset & IDs & Views & Conditions & Train & Calib. & Test & Gallery & Probes & \#Gallery & \#Probes \\',
         r'\midrule'] + rows + [r'\bottomrule', r'\end{tabular}']
    write('t_datasets', '\n'.join(L) + '\n')


def t_reproduction():
    R = J('results/p5_frozen/results.json')
    R.update(J('results/p5_frozen/repro.json'))             # GaitPart, GaitGL (frozen_hybrid repro)
    pub = {'gaitset': (r'GaitSet~\cite{chao2019gaitset}', (95.8, 90.0, 75.4)),
           'gaitpart': (r'GaitPart~\cite{fan2020gaitpart}', (96.1, 90.7, 78.7)),
           'gaitgl': (r'GaitGL~\cite{lin2021gaitgl}', (97.4, 94.5, 83.8)),
           'gaitbase': (r'GaitBase~\cite{fan2023opengait}', (97.6, 94.0, 77.4))}
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
    L = [r'\begin{tabular}{lrcc}', r'\toprule',
         r'Model & Parameters & Enrollment (ms/identity, $K{=}4$) & Matching (ms/probe, 50 IDs) \\', r'\midrule']
    for k, lab in [('B1', 'Compact backbone (B1)'), ('B1+head', 'B1 + reliability head')]:
        L.append(f"{lab} & {T[k]['params']:,} & {T[k]['enroll_ms_per_identity_K4']:.2f} & "
                 f"{T[k]['match_ms_per_probe_50way']:.2f}" + r' \\')
    L += [r'\midrule', r'Head on GaitPart, GaitGL / GaitSet, GaitBase & +8.4k / +16.5k & \multicolumn{2}{c}{$<$1\% of backbone parameters} \\',
          r'\bottomrule', r'\end{tabular}']
    write('t_complexity', '\n'.join(L) + '\n')


T_NOTATION = r"""\begin{tabular}{ll}
\toprule
Symbol & Meaning \\
\midrule
$x$, $q$ & input sample (silhouette sequence or GEI); probe sample \\
$P$, $D$ & number of horizontal body parts; part embedding dimension \\
$\boldsymbol{\mu}_i(x)\in\mathbb{R}^D$ & frozen embedding of part $i$ of sample $x$ \\
$s_i(x)>0$ & predicted variance (unreliability) of part $i$ of sample $x$ \\
$h_\phi$ & reliability head with parameters $\phi$ \\
$\mathcal{E}_n=\{x_{n,1},\dots,x_{n,K}\}$ & enrollment set of identity $n$ ($K$ samples) \\
$\boldsymbol{c}_{n,i}$, $s_i(c_n)$ & fused prototype mean and variance of part $i$ of identity $n$ \\
$\ell(q,c_n)$, $\ell_i(q,c_n)$ & matching log-likelihood and its part-$i$ term \\
$\omega_i(q,c_n)$ & effective part weight $1/(s_i(q)+s_i(c_n))$ \\
$\Delta_i(q)$ & contribution of part $i$ to the decision margin \\
$\mathcal{C}$, $\mathcal{G}$ & calibration identities; enrolled gallery identities \\
$\tau$ & open-set acceptance threshold \\
\bottomrule
\end{tabular}
"""

T_HPARAMS = r"""\begin{tabular}{lll}
\toprule
Component & Hyperparameter & Value \\
\midrule
Compact backbone (B1) & architecture & 3 conv.\ blocks (32/64/128 ch.) with BN, $P=8$, $D=128$ \\
 & episodes & 20-way, 4-shot, 5 queries, cross-view, $\leq$8{,}000 \\
 & optimiser & Adam, lr $10^{-3}$, wd $10^{-5}$, cosine schedule, clip 1.0 \\
 & per-part auxiliary loss weight & 1.0 \\
 & early stopping & every 200 episodes, patience 12 \\
\midrule
Reliability head & architecture & MLP $2D\!\to\!64\!\to\!1$, shared, + per-part bias \\
 & variance & softplus, floor $10^{-3}$, initial value 4 \\
 & calibration & 3{,}000 episodes, $N=\min(20,|\mathcal{C}|-2)$, $K=4$, 5 probes \\
 & optimiser & Adam, lr $10^{-3}$ \\
 & occlusion-aware sampling & horizontal band, 6--16 rows, $p=0.5$ \\
\midrule
Evaluation & seeds & 5 (CASIA-B), 3 (CASIA-E, OU-MVLP) \\
 & open-set splits & 10 / 5 / 2, half of the test identities enrolled \\
\bottomrule
\end{tabular}
"""

T_POSITIONING = r"""\begin{tabular}{lcccccc}
\toprule
Approach & Frozen & Per- & Pair- & Calibrated & Open-set & Exact \\
 & backbone & part & dependent & uncertainty & score & explanation \\
\midrule
Part-based gait models~\cite{chao2019gaitset,fan2020gaitpart,fan2023opengait} & -- & \checkmark & -- & -- & -- & -- \\
Adaptive region weighting~\cite{p14,p18} & -- & \checkmark & -- & -- & -- & -- \\
Few-shot metric heads~\cite{snell2017prototypical,oreshkin2018tadam,ye2020feat} & -- & -- & partly & -- & -- & -- \\
Probabilistic face embeddings~\cite{shi2019pfe,chang2020dul} & \checkmark & -- & \checkmark & \checkmark & -- & -- \\
\midrule
\textbf{Ours} & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark \\
\bottomrule
\end{tabular}
"""


T_GUIDANCE = r"""\begin{tabular}{lll}
\toprule
Operating condition & Component & Evidence \\
\midrule
Probes partly occluded & per-part likelihood (N1) & Tables~\ref{tab:multi}, \ref{tab:baselines} \\
Enrollment walks partly occluded & precision-weighted enrollment (N2), $K>1$ & Fig.~\ref{fig:severity}, Remark~\ref{rem:reject} \\
Body regions leaving the field of view & silhouette-energy prior (Eq.~\ref{eq:hybrid}) & Table~\ref{tab:baselines} \\
Frozen backbone, occlusion-heavy site & precision-weighted backbone distance (Eq.~\ref{eq:wdist}) & Table~\ref{tab:frozenwdist} \\
Clothing and carrying covariates & per-part likelihood (N1) & Tables~\ref{tab:multi}, \ref{tab:frozen} \\
Unknown walkers & likelihood-based rejection (N4) & Fig.~\ref{fig:dirfar}, Table~\ref{tab:frozen} \\
Auditing a decision & margin decomposition (Eqs.~\ref{eq:margin}, \ref{eq:margin_split}) & Table~\ref{tab:faith} \\
Existing deployed backbone & plug-in head, no retraining & Tables~\ref{tab:frozen}, \ref{tab:frozentypes} \\
\bottomrule
\end{tabular}
"""


def t_static():
    write('t_guidance', T_GUIDANCE)
    write('t_notation', T_NOTATION)
    write('t_hparams', T_HPARAMS)
    write('t_positioning', T_POSITIONING)


# ------------------------------------------------------------------ figures
def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.3, 'legend.frameon': False})
    return plt


COL = {'b1': '#7f7f7f', 'head': '#1f77b4', 'frozen': '#7f7f7f'}
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
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.8), sharey=False)
    for a, (k, t) in zip(ax, [('nm', 'NM probes'), ('bg', 'BG probes'), ('cl', 'CL probes'), ('nm_occ', 'Occluded NM probes')]):
        b = np.array([R[f'b1/{s}'][k] for s in range(5)]); h = np.array([R[f'head/{s}'][k] for s in range(5)])
        x = np.arange(11)
        for arr, lab, c, m in [(b, 'B1', COL['b1'], 's'), (h, 'B1 + head', COL['head'], 'o')]:
            a.plot(x, arr.mean(0), marker=m, ms=3.5, color=c, label=lab)
            a.fill_between(x, arr.min(0), arr.max(0), color=c, alpha=0.15, lw=0)
        a.set_xticks(x); a.set_xticklabels(VIEWS, fontsize=7); a.set_title(t); a.set_xlabel('probe view (deg)')
    ax[0].set_ylabel('rank-1 (%)'); ax[0].legend(loc='lower center')
    plt.tight_layout(); plt.savefig(f'{FIG}/per_view.pdf'); plt.close()


def f_severity():
    plt = _style()
    A = J('results/paper_analyses.json')
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.9))
    for a, (sec, xs, xl, t) in zip(ax[:2], [('enroll_sweep', ['0.0', '0.25', '0.5', '0.75', '1.0'],
                                             'fraction of occluded enrollment samples', '(a) Corrupted enrollment'),
                                            ('probe_sweep', ['0', '8', '16', '24', '32'],
                                             'occluder height on probe (rows of 64)', '(b) Occluded probes')]):
        R = A[sec]
        for kind, lab, m in [('b1', 'B1', 's'), ('head', 'B1 + head', 'o')]:
            arr = np.array([[R[f'{kind}/{s}'][x] for x in xs] for s in range(5)])
            xv = [float(x) for x in xs]
            a.errorbar(xv, arr.mean(0), yerr=arr.std(0), marker=m, ms=4, capsize=2, color=COL[kind], label=lab)
        a.set_xlabel(xl); a.set_title(t)
    ax[0].set_ylabel('rank-1 (%)'); ax[0].legend()
    R = A['kshot']; a = ax[2]
    for kind, lab, c in [('b1', 'B1', COL['b1']), ('head', 'B1 + head', COL['head'])]:
        for g, ls, m in [('clean', '-', 'o'), ('corrupt', '--', '^')]:
            arr = np.array([[R[f'{kind}/{s}'][f'{g}/K{k}'] for k in [1, 2, 3, 4]] for s in range(5)])
            a.plot([1, 2, 3, 4], arr.mean(0), ls=ls, marker=m, ms=4, color=c, label=f'{lab}, {g}')
    a.set_xticks([1, 2, 3, 4]); a.set_xlabel('enrollment samples $K$'); a.set_title('(c) Enrollment size'); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f'{FIG}/severity.pdf'); plt.close()


def f_dirfar():
    plt = _style()
    D = J('results/paper_analyses.json')['dirfar']
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.9), sharey=True)
    for a, (m, t) in zip(ax, [('nm', 'NM probes'), ('cl', 'CL probes'), ('nm_occ', 'Occluded NM probes')]):
        for kind, lab in [('b1', 'B1'), ('head', 'B1 + head')]:
            fa, dr = D[kind][m]
            a.plot(np.array(fa) * 100, np.array(dr) * 100, color=COL[kind], label=lab, lw=1.8)
        a.set_xscale('log'); a.set_xlabel('false alarm rate (%)'); a.set_title(t)
    ax[0].set_ylabel('detection and identification rate (%)'); ax[0].legend(loc='upper left')
    plt.tight_layout(); plt.savefig(f'{FIG}/dirfar.pdf'); plt.close()


def f_reliability():
    plt = _style()
    R = J('results/paper_analyses.json')['reliability']
    fig, a = plt.subplots(1, 1, figsize=(3.6, 3.2))
    allv = []
    for s, r in R.items():
        a.plot(r['pred'], r['obs'], marker='o', ms=3, color=COL['head'], alpha=0.6, lw=1)
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
    fig, a = plt.subplots(1, 1, figsize=(7.5, 2.9))
    w = 0.26
    for j, (ds, lab, c) in enumerate([('casiab', 'CASIA-B', '#1f77b4'), ('casiae', 'CASIA-E', '#ff7f0e'), ('oumvlp', 'OU-MVLP', '#2ca02c')]):
        R = J(f'results/md_eval_{ds}.json')
        seeds = [s for s in R if 'b1' in R[s] and 'head' in R[s]]
        d = [[metrics(R[s]['head'])[k] - metrics(R[s]['b1'])[k] for s in seeds] for k, _ in keys]
        a.bar(np.arange(len(keys)) + (j - 1) * w, [np.mean(x) for x in d], w, yerr=[np.std(x) for x in d],
              capsize=2, color=c, label=lab)
    a.axhline(0, color='k', lw=0.8); a.set_xticks(np.arange(len(keys))); a.set_xticklabels([k[1] for k in keys])
    a.set_ylabel('gain over B1 (points)'); a.legend(ncol=3, loc='upper right')
    plt.tight_layout(); plt.savefig(f'{FIG}/robustness_bars.pdf'); plt.close()


def f_heads_bars():
    plt = _style()
    rows = [('Proto', 'results/p3_bn'), ('Matching', 'results/p5_matching'), ('Relation', 'results/p5_relation'),
            ('Meta-B.', 'results/p5_metabase'), ('FEAT', 'results/p5_feat'), ('TADAM', 'results/p5_tadam'),
            ('B1', 'results/p3_bn_aux')]
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
    names = [r[0] for r in rows] + ['Ours']
    m = [J(f'{d}/summary.json')['evals']['full_gallery_cv'] for _, d in rows]
    ours = J('results/p4_2stage_occ/summary.json')['evals']['fgcv_n1n2_lam1']
    for a, key, t in [(ax[0], 'mean', 'Cross-view rank-1 (mean)'), (ax[1], 'cl', 'Cross-view rank-1 (CL)')]:
        vals = [x[key]['mean'] for x in m] + [ours[key]['mean']]
        errs = [x[key]['ci95'] for x in m] + [ours[key]['ci95']]
        cols = ['#bbbbbb'] * (len(rows) - 1) + ['#7f7f7f', '#1f77b4']
        a.bar(names, vals, yerr=errs, capsize=2, color=cols); a.set_title(t)
        a.tick_params(axis='x', labelsize=8); a.set_ylim(min(vals) - 8, max(vals) + 4)
    ax[0].set_ylabel('rank-1 (%)')
    plt.tight_layout(); plt.savefig(f'{FIG}/heads_bars.pdf'); plt.close()


def f_attention_bars():
    plt = _style()
    rows = [('Uniform', 'abl_cv_uniform'), ('Global', 'abl_cv_global'), ('Query', 'abl_cv_query'),
            ('[q; c̄]', 'abl_cv_concat'), ('Residual', 'v0_dppnet_lt_cv')]
    fig, a = plt.subplots(1, 1, figsize=(4.6, 2.8))
    x = np.arange(len(rows)); w = 0.26
    for j, (c, col) in enumerate([('nm', '#9ecae1'), ('bg', '#4292c6'), ('cl', '#08519c')]):
        v = [J(f'results/{r}/summary.json')['evals']['full_gallery_cv'][c] for _, r in rows]
        a.bar(x + (j - 1) * w, [z['mean'] for z in v], w, yerr=[z['ci95'] for z in v], capsize=2, color=col, label=c.upper())
    a.set_xticks(x); a.set_xticklabels([r[0] for r in rows], fontsize=8); a.set_ylabel('cross-view rank-1 (%)')
    a.legend(ncol=3, fontsize=7)
    plt.tight_layout(); plt.savefig(f'{FIG}/attention_bars.pdf'); plt.close()


def f_frozen_types():
    plt = _style()
    from .latex_tables import FROZEN_BBS, frozen_occ_types
    X = frozen_occ_types()
    types = [('clean', 'Clean'), ('hband', 'H. band$^\\dagger$'), ('bottom', 'Lower body'), ('vband', 'V. band'), ('noise', 'Noise')]
    bbs = [(bb, f'{n} (frozen)') for bb, n, _ in FROZEN_BBS if bb in X]
    fig, ax = plt.subplots(2, 2, figsize=(9, 5.2), sharey=True)
    ax = ax.ravel()
    for a, (bb, t) in zip(ax, bbs):
        fr = [np.mean([d['frozen'][k] for d in X[bb].values()]) for k, _ in types]
        hd = [np.mean([r[k] for d in X[bb].values() for r in d['head'].values()]) for k, _ in types]
        x = np.arange(len(types))
        a.bar(x - 0.2, fr, 0.4, color=COL['frozen'], label='frozen'); a.bar(x + 0.2, hd, 0.4, color=COL['head'], label='+ head')
        a.set_xticks(x); a.set_xticklabels([t_[1] for t_ in types], fontsize=8); a.set_title(t)
    ax[0].set_ylabel('rank-1 (%)'); ax[2].set_ylabel('rank-1 (%)'); ax[0].legend()
    plt.tight_layout(); plt.savefig(f'{FIG}/frozen_types.pdf'); plt.close()


def f_baselines():
    """Robustness baselines vs the head and the hybrid, per dataset (results/robust_baselines_*.json)."""
    plt = _style()
    kinds = [('b1', 'Mean (B1)', '#c7c7c7'), ('median', 'Median', '#a6a6a6'), ('energy', 'Energy mask', '#7f7f7f'),
             ('head_global', 'Pooled var.', '#9ecae1'), ('head', 'Head (ours)', '#1f77b4'),
             ('hybrid', 'Head + energy (ours)', '#08306b')]
    scen = [('corrupt_enroll', 'Corrupted\nenrollment'), ('occ_hband', 'Horizontal\nband$^\\dagger$'),
            ('occ_bottom', 'Lower\nbody')]
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.9))
    w = 0.13
    for a, (ds, t) in zip(ax, [('casiab', 'CASIA-B'), ('casiae', 'CASIA-E'), ('oumvlp', 'OU-MVLP')]):
        R = J(f'results/robust_baselines_{ds}.json')
        seeds = sorted(R)
        sc = [s for s in scen if any(c.split('>')[0] == s[0] for c in R[seeds[0]]['acc'])]
        for j, (k, lab, col) in enumerate(kinds):
            v = [[np.mean([x[k] for c, x in R[s]['acc'].items() if c.split('>')[0] == s_]) for s in seeds]
                 for s_, _ in sc]
            a.bar(np.arange(len(sc)) + (j - 2.5) * w, [np.mean(x) for x in v], w, yerr=[np.std(x) for x in v],
                  capsize=1.5, color=col, label=lab)
        a.set_xticks(np.arange(len(sc))); a.set_xticklabels([s[1] for s in sc]); a.set_title(t)
    ax[0].set_ylabel('rank-1 (%)')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.02), fontsize=8)
    plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(f'{FIG}/baselines.pdf'); plt.close()



if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    for f in [t_datasets, t_reproduction, t_static, f_robustness_bars, f_heads_bars, f_attention_bars, f_frozen_types,
              f_baselines]:
        f(); print('ok', f.__name__)
    A = J('results/paper_analyses.json')
    if 'timing' in A:
        for f in [t_complexity, f_per_view, f_severity, f_dirfar, f_reliability]:
            f(); print('ok', f.__name__)
    else:
        print('paper_analyses still running: per-view / severity / dirfar / reliability / complexity pending')
