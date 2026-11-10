"""Protocol-integrity tests. Run: PYTHONPATH=src python -m pytest tests -q"""
import os

import numpy as np
import pytest

from gaitlab.data.casiab import CasiaB, LT_GALLERY, LT_PROBES, VIEWS, SEQS
from gaitlab.data.sampler import EpisodeSampler

CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache', 'casiab_gei.npz')


@pytest.fixture(scope='module')
def ds():
    if not os.path.exists(CACHE):
        pytest.skip('CASIA-B cache not built (see README)')
    return CasiaB(CACHE)


@pytest.fixture(scope='module')
def split(ds):
    return ds.lt_split()


def test_split_is_standard_lt(ds, split):
    all_ = split['train'] + split['val'] + split['test']
    assert len(set(all_)) == len(all_) == 124
    assert all(int(ds.subjects[i]) > 74 for i in split['test'])
    assert all(int(ds.subjects[i]) <= 74 for i in split['train'] + split['val'])
    assert len(split['test']) == 50


@pytest.mark.parametrize('cond', ['nm', 'bg', 'cl'])
@pytest.mark.parametrize('same_view', [True, False])
def test_support_query_disjoint_and_valid(ds, split, cond, same_view):
    smp = EpisodeSampler(ds, split['test'], LT_GALLERY, LT_PROBES[cond],
                         VIEWS, VIEWS, n_way=20, k_shot=4, n_query=2,
                         same_view=same_view, rng=np.random.RandomState(1))
    for _ in range(200):
        sup, qry, lab, view = smp.sample()
        assert len(set(sup)) == len(sup), 'duplicate support cells'
        assert not set(sup) & set(qry), 'support/query overlap'
        for c in sup + qry:
            assert ds.valid[c], 'invalid (missing/short) cell sampled'
        for c in sup:
            assert SEQS[c[1]] in LT_GALLERY
        for c in qry:
            assert SEQS[c[1]] in LT_PROBES[cond]
        # every query's identity equals its label's support identity
        sup_ids = [sup[i * 4][0] for i in range(20)]
        assert all(sup_ids[l] == c[0] for c, l in zip(qry, lab))
        if same_view:
            assert {c[2] for c in sup + qry} == {view}
        else:
            for c, l in zip(qry, lab):
                assert c[2] not in {s[2] for s in sup[l * 4:(l + 1) * 4]}


def test_k_larger_than_available_is_rejected(ds, split):
    smp = EpisodeSampler(ds, split['test'], LT_GALLERY, LT_PROBES['nm'],
                         VIEWS, VIEWS, n_way=5, k_shot=5, n_query=1,
                         same_view=True)
    with pytest.raises(RuntimeError):
        smp.sample()


def test_tensor_matches_cell(ds, split):
    smp = EpisodeSampler(ds, split['test'], LT_GALLERY, LT_PROBES['bg'],
                         VIEWS, VIEWS, n_way=5, k_shot=2, n_query=1)
    sup, qry, _, _ = smp.sample()
    t = smp.to_tensors(qry)
    s, q, v = qry[0]
    assert np.allclose(t[0, 0].numpy() * 255, ds.gei[s, q, v], atol=0.5)
