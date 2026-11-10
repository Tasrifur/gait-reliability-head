"""Episode sampler with guaranteed support/query disjointness.

A "cell" is one GEI, identified by (subject, seq, view). For each sampled
identity, support cells are drawn WITHOUT replacement from the valid gallery
cells, and query cells are drawn from the valid probe cells minus any cell
used as support. If an identity cannot supply K distinct support cells and at
least one query cell, it is not eligible for that episode configuration.
"""
import numpy as np


class EpisodeSampler:
    def __init__(self, ds, subjects, gallery_seqs, probe_seqs,
                 gallery_views, probe_views, n_way, k_shot, n_query,
                 same_view=True, rng=None, query_replace=False):
        """
        ds            : CasiaB
        subjects      : list of subject indices to sample identities from
        gallery_seqs  : seq names usable as support (e.g. LT_GALLERY)
        probe_seqs    : seq names usable as queries
        gallery_views : view names usable for support
        probe_views   : view names usable for queries
        same_view     : if True, each episode fixes one view (shared by
                        support and query); probe_views is ignored.
                        If False, support views and query views are drawn
                        independently and the query view is never equal
                        to any support view (identical view excluded).
        query_replace : allow repeated query cells when an identity has fewer
                        than n_query distinct probe cells (default False:
                        use all available instead).
        """
        self.ds, self.subjects = ds, list(subjects)
        self.gq = [ds.qidx(q) for q in gallery_seqs]
        self.pq = [ds.qidx(q) for q in probe_seqs]
        self.gv = [ds.vidx(v) for v in gallery_views]
        self.pv = [ds.vidx(v) for v in probe_views]
        self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
        self.same_view = same_view
        self.rng = rng or np.random.RandomState(0)
        self.query_replace = query_replace

    def _cells(self, s, seqs, views):
        v = self.ds.valid
        return [(s, q, w) for q in seqs for w in views if v[s, q, w]]

    def _try_identity(self, s, view):
        if self.same_view:
            g = self._cells(s, self.gq, [view])
            p = self._cells(s, self.pq, [view])
        else:
            g = self._cells(s, self.gq, self.gv)
            p = None
        if len(g) < self.k_shot:
            return None
        sup = [g[i] for i in self.rng.choice(len(g), self.k_shot, replace=False)]
        if self.same_view:
            p = [c for c in p if c not in sup]
        else:
            used_views = {c[2] for c in sup}
            p = [c for c in self._cells(s, self.pq, self.pv)
                 if c[2] not in used_views and c not in sup]
        if not p:
            return None
        if len(p) >= self.n_query:
            qry = [p[i] for i in self.rng.choice(len(p), self.n_query, replace=False)]
        elif self.query_replace:
            qry = [p[i] for i in self.rng.choice(len(p), self.n_query, replace=True)]
        else:
            qry = list(p)
        return sup, qry

    def sample(self, view=None):
        """Returns (support_cells, query_cells, query_labels, view).
        support_cells: list of N*K cells ordered class-major.
        Retries identities until N eligible ones are found."""
        if self.same_view and view is None:
            view = self.gv[self.rng.randint(len(self.gv))]
        order = self.rng.permutation(self.subjects)
        sup_all, qry_all, labels = [], [], []
        for s in order:
            r = self._try_identity(int(s), view)
            if r is None:
                continue
            sup, qry = r
            c = len(sup_all) // self.k_shot
            sup_all += sup
            qry_all += qry
            labels += [c] * len(qry)
            if c + 1 == self.n_way:
                break
        if len(sup_all) // self.k_shot < self.n_way:
            raise RuntimeError(
                f'only {len(sup_all)//self.k_shot} eligible identities for '
                f'{self.n_way}-way (view={view})')
        return sup_all, qry_all, np.array(labels), view

    def to_tensors(self, cells):
        import torch
        return torch.from_numpy(np.stack([self.ds.get(*c) for c in cells]))


class LegacySampler(EpisodeSampler):
    """Reproduces the ORIGINAL notebook sampler (ViewAwareEpisodeDataset /
    eval_accuracy) for measuring protocol inflation only. Do not use for
    reported results. Behaviour copied from the notebook:
      * one random view per episode (same-view);
      * support: k_shot draws WITH replacement from gallery seqs at that view;
      * query:   n_query draws WITH replacement from probe seqs at that view,
                 with no exclusion of support cells (overlap possible).
    Missing cells are skipped (the notebook tensor padded them instead).
    """

    def _try_identity(self, s, view):
        g = self._cells(s, self.gq, [view])
        p = self._cells(s, self.pq, [view])
        if not g or not p:
            return None
        sup = [g[i] for i in self.rng.choice(len(g), self.k_shot, replace=True)]
        qry = [p[i] for i in self.rng.choice(len(p), self.n_query, replace=True)]
        return sup, qry
