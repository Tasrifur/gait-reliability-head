"""CASIA-B GEI dataset with explicit (subject, seq, view) indexing.

All indexing goes through named seqs/views; there is no flat-index arithmetic.
"""
import numpy as np

SEQS = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06',
        'bg-01', 'bg-02', 'cl-01', 'cl-02']
VIEWS = ['000', '018', '036', '054', '072', '090',
         '108', '126', '144', '162', '180']
COND_SEQS = {'nm': SEQS[:6], 'bg': SEQS[6:8], 'cl': SEQS[8:]}

# Standard LT protocol (Wu et al. / GaitSet): gallery NM-01..04, probes below.
LT_GALLERY = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
LT_PROBES = {'nm': ['nm-05', 'nm-06'], 'bg': ['bg-01', 'bg-02'],
             'cl': ['cl-01', 'cl-02']}


class CasiaB:
    def __init__(self, path, min_frames=15):
        z = np.load(path)
        assert list(z['seqs']) == SEQS and list(z['views']) == VIEWS
        self.gei = z['gei']                      # uint8 [S, 10, 11, 64, 64]
        self.nframes = z['nframes']
        self.subjects = [str(s) for s in z['subjects']]
        self.valid = self.nframes >= min_frames  # [S, 10, 11] bool
        self.min_frames = min_frames
        self.views, self.seqs = list(VIEWS), list(SEQS)   # same interface as DenseGEI

    # --- index helpers -------------------------------------------------------
    def sidx(self, subject):
        return self.subjects.index(subject)

    @staticmethod
    def qidx(seq):
        return SEQS.index(seq)

    @staticmethod
    def vidx(view):
        return VIEWS.index(view)

    def get(self, s, q, v):
        """Float GEI in [0,1], shape [1, 64, 64]; s/q/v are integer indices."""
        return self.gei[s, q, v][None].astype(np.float32) / 255.0

    # --- splits --------------------------------------------------------------
    def lt_split(self, n_val=10, seed=0):
        """Standard LT: subjects 001-074 train, 075-124 test.
        `n_val` training subjects (chosen with `seed`) are held out for
        model selection. Returns dict of subject-index lists."""
        train = [i for i, s in enumerate(self.subjects) if int(s) <= 74]
        test = [i for i, s in enumerate(self.subjects) if int(s) > 74]
        rng = np.random.RandomState(seed)
        val = sorted(rng.choice(train, n_val, replace=False).tolist())
        train = [i for i in train if i not in val]
        return {'train': train, 'val': val, 'test': test}

    def legacy_split(self):
        """The random 74/25/25 split used by the original notebooks
        (np.random.seed(42); permutation of 124). For comparison only."""
        np.random.seed(42)
        perm = np.random.permutation(len(self.subjects))
        return {'train': perm[:74].tolist(), 'val': perm[74:99].tolist(),
                'test': perm[99:].tolist()}
