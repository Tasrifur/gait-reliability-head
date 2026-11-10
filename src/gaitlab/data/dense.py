"""Dense GEI datasets for OU-MVLP and CASIA-E with the same interface as CasiaB
(gei[subject, seq, view], valid mask, qidx/vidx/get), so EpisodeSampler and the
training code work unchanged.

OU-MVLP : seqs ['00', '01'] (one walk each), 14 views.
          Official split: train = odd ids (5,153), test = even ids (5,154).
CASIA-E : seq axis = '{height}-{scene}-{cond}-{n}' with height H|L,
          scene in scene1|scene2|scene3|scene3s, cond nm|bg|cl, n in 1|2 (48 entries);
          seq 3/4 (a second camera, only at 045/135) are excluded, as in the
          official protocol. 13 views (270 excluded).
          Official split: train = 001-200, test = 201-1014.
Held-out calibration identities: a fixed subset of the official TRAINING ids is
removed from backbone training and used for model selection (validation),
reliability-head calibration and open-set rule selection.

Usage (build caches): python -m gaitlab.data.dense
"""
import numpy as np

OU_SEQS = ['00', '01']
OU_VIEWS = ['000', '015', '030', '045', '060', '075', '090', '180', '195', '210', '225', '240', '255', '270']
CE_VIEWS = ['000', '015', '030', '045', '060', '075', '090', '105', '120', '135', '150', '165', '180']
CE_SEQS = [f'{h}-{sc}-{c}-{n}' for h in ['H', 'L'] for sc in ['scene1', 'scene2', 'scene3', 'scene3s']
           for c in ['nm', 'bg', 'cl'] for n in [1, 2]]


def build(name):
    import pandas as pd                          # only needed to build the caches
    if name == 'oumvlp':
        X = np.load('data/cache/oumvlp_gei.npy', mmap_mode='r')
        m = pd.read_csv('data/cache/oumvlp_meta.csv', dtype=str)
        m['seqname'] = m.seq.str.zfill(2)
        seqs, views = OU_SEQS, OU_VIEWS
    else:
        X = np.load('data/cache/casiae_gei.npy', mmap_mode='r')
        m = pd.read_csv('data/cache/casiae_meta.csv', dtype=str).fillna('')
        m = m[m.seq.isin(['1', '2']) & m.angle.isin(CE_VIEWS)]
        m['seqname'] = m.height + '-' + m.scene + m.variant + '-' + m.cond + '-' + m.seq
        seqs, views = CE_SEQS, CE_VIEWS
    m = m[m.angle.isin(views)]
    subjects = sorted(m.subject.unique())
    si = {s: i for i, s in enumerate(subjects)}
    qi = {q: i for i, q in enumerate(seqs)}
    vi = {v: i for i, v in enumerate(views)}
    gei = np.zeros((len(subjects), len(seqs), len(views), 64, 64), np.uint8)
    valid = np.zeros((len(subjects), len(seqs), len(views)), bool)
    rows = m.index.values
    order = np.argsort(rows)                     # sequential mmap reads
    for k in order:
        r = m.iloc[k]
        a, b, c = si[r.subject], qi[r.seqname], vi[r.angle]
        gei[a, b, c] = X[rows[k]]
        valid[a, b, c] = True
    np.savez(f'data/cache/{name}_dense.npz', gei=gei, valid=valid, subjects=np.array(subjects),
             seqs=np.array(seqs), views=np.array(views))
    print(name, gei.shape, 'valid', int(valid.sum()))


class DenseGEI:
    def __init__(self, name):
        z = np.load(f'data/cache/{name}_dense.npz')
        self.name = name
        self.gei, self.valid = z['gei'], z['valid']
        self.subjects = [str(s) for s in z['subjects']]
        self.seqs, self.views = [str(s) for s in z['seqs']], [str(v) for v in z['views']]
        self._q = {q: i for i, q in enumerate(self.seqs)}
        self._v = {v: i for i, v in enumerate(self.views)}

    def qidx(self, q):
        return self._q[q]

    def vidx(self, v):
        return self._v[v]

    def get(self, s, q, v):
        return self.gei[s, q, v][None].astype(np.float32) / 255.0

    # named sequence sets used by configs
    def seq_set(self, name):
        if self.name == 'oumvlp':
            return {'all': self.seqs, 'gallery': ['01'], 'probe': ['00'], 'nm_all': self.seqs}[name]
        nm = [q for q in self.seqs if '-nm-' in q]
        return {'all': self.seqs, 'nm_all': nm,
                'gallery': [q for q in nm if '-scene1-' in q]}[name]

    def probe_sets(self):
        if self.name == 'oumvlp':
            return {'nm': ['00']}
        return {c: [q for q in self.seqs if f'-{c}-' in q and '-scene1-' not in q] for c in ['nm', 'bg', 'cl']}

    def split(self, n_cal, seed=0):
        """Official train/test; n_cal training ids held out for validation/calibration."""
        ids = np.array([int(s) for s in self.subjects])
        if self.name == 'oumvlp':
            train = [i for i, x in enumerate(ids) if x % 2 == 1]
            test = [i for i, x in enumerate(ids) if x % 2 == 0]
        else:
            train = [i for i, x in enumerate(ids) if x <= 200]
            test = [i for i, x in enumerate(ids) if x > 200]
        rng = np.random.RandomState(seed)
        cal = sorted(rng.choice(train, n_cal, replace=False).tolist())
        return {'train': [i for i in train if i not in set(cal)], 'val': cal, 'test': test}


if __name__ == '__main__':
    for n in ['oumvlp', 'casiae']:
        build(n)
