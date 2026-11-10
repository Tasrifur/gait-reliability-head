"""Evaluation for CASIA-B: episodic N-way K-shot and full-gallery rank-1.

Both use disjoint gallery/probe sequences (LT protocol by default).
"""
import numpy as np
import torch

from .data.casiab import LT_GALLERY, LT_PROBES, VIEWS
from .data.casiab import COND_SEQS
from .data.sampler import EpisodeSampler, LegacySampler
from .utils import mean_ci

LEGACY_GALLERY = ['nm-05', 'nm-06']


@torch.no_grad()
def episodic(model, ds, subjects, cond, n_way=20, k_shot=4, n_query=2,
             same_view=True, episodes=500, seed=12345, device='cuda',
             gallery_seqs=LT_GALLERY, probe_seqs=None, views=VIEWS, legacy=False):
    """Per-episode CCR. Returns dict(mean, ci95, sd, n, per_episode).
    legacy=True reproduces the original notebook evaluation (gallery NM-05/06
    with replacement, NM probes from NM-01..06 incl. gallery)."""
    model.eval()
    if legacy:
        smp = LegacySampler(ds, subjects, LEGACY_GALLERY, COND_SEQS[cond], views, views,
                            n_way, k_shot, n_query, same_view=True,
                            rng=np.random.RandomState(seed))
    else:
        smp = EpisodeSampler(ds, subjects, gallery_seqs, probe_seqs or LT_PROBES[cond],
                             views, views, n_way, k_shot, n_query,
                             same_view=same_view, rng=np.random.RandomState(seed))
    accs = []
    for _ in range(episodes):
        sup, qry, lab, _ = smp.sample()
        S = smp.to_tensors(sup).to(device)
        Q = smp.to_tensors(qry).to(device)
        logits, _ = model(S, Q, n_way, k_shot)
        accs.append((logits.argmax(1).cpu().numpy() == lab).mean() * 100)
    r = mean_ci(accs)
    r['per_episode'] = [float(a) for a in accs]
    return r


@torch.no_grad()
def full_gallery_rank1(model, ds, subjects, cond, cross_view=True, device='cuda',
                       gallery_seqs=LT_GALLERY):
    """Standard CASIA-B evaluation: every test subject is in the gallery.

    For each (gallery view, probe view) pair, each subject's prototype is the
    mean of its valid gallery GEIs at the gallery view; every valid probe GEI
    at the probe view is classified among all subjects present.
    cross_view=True excludes identical views (the published LT numbers);
    False evaluates only identical views.
    Returns {'per_probe_view': {view: acc}, 'mean': acc}.
    """
    model.eval()
    gq = [ds.qidx(q) for q in gallery_seqs]
    pq = [ds.qidx(q) for q in LT_PROBES[cond]]
    per_view = {}
    for pv in range(len(VIEWS)):
        accs = []
        for gv in range(len(VIEWS)):
            if (gv == pv) == cross_view:
                continue
            ids = [s for s in subjects if ds.valid[s, gq, gv].any()]
            sup = [(s, q, gv) for s in ids for q in gq if ds.valid[s, q, gv]]
            # unequal shot counts: encode all, average per subject
            X = torch.from_numpy(np.stack([ds.get(*c) for c in sup])).to(device)
            owner = torch.tensor([ids.index(c[0]) for c in sup], device=device)
            qry = [(s, q, pv) for s in ids for q in pq if ds.valid[s, q, pv]]
            if not qry:
                continue
            Q = torch.from_numpy(np.stack([ds.get(*c) for c in qry])).to(device)
            lab = np.array([ids.index(c[0]) for c in qry])
            logits = model.logits_unequal(X, owner, len(ids), Q)
            accs.append((logits.argmax(1).cpu().numpy() == lab).mean() * 100)
        per_view[VIEWS[pv]] = float(np.mean(accs))
    return {'per_probe_view': per_view, 'mean': float(np.mean(list(per_view.values())))}
