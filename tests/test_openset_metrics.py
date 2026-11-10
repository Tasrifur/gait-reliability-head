"""Open-set metric checks. Run: PYTHONPATH=src python -m pytest tests -q"""
import numpy as np
from sklearn.metrics import roc_auc_score

from gaitlab.eval_openset import auroc, dir_at_far, oscr


def _data(seed=0, ties=False):
    r = np.random.RandomState(seed)
    sk = r.normal(1.0, 1.0, 400)
    su = r.normal(0.0, 1.0, 300)
    if ties:
        sk, su = np.round(sk, 1), np.round(su, 1)
    ck = r.rand(400) < 0.8
    return sk, ck, su


def test_auroc_matches_sklearn_with_ties():
    for ties in (False, True):
        sk, _, su = _data(1, ties)
        y = np.r_[np.ones(len(sk)), np.zeros(len(su))]
        assert abs(auroc(sk, su) - roc_auc_score(y, np.r_[sk, su])) < 1e-9


def test_dir_at_far_bruteforce():
    sk, ck, su = _data(2)
    for far in (0.01, 0.05, 0.1):
        t = np.quantile(su, 1 - far, method='higher')
        assert (su > t).mean() <= far + 1e-12
        assert dir_at_far(sk, ck, su, far) == ((sk > t) & ck).mean()


def test_oscr_bruteforce_and_bounds():
    sk, ck, su = _data(3)
    ts = np.unique(np.r_[sk, su])[::-1]
    ccr = [0] + [((sk >= t) & ck).mean() for t in ts] + [ck.mean()]
    fpr = [0] + [(su >= t).mean() for t in ts] + [1.0]
    assert abs(oscr(sk, ck, su) - np.trapz(ccr, fpr)) < 1e-9
    assert 0 <= oscr(sk, ck, su) <= ck.mean()
    # perfect separation + all correct -> OSCR = 1
    assert abs(oscr(np.ones(10) * 5, np.ones(10, bool), np.zeros(10)) - 1.0) < 1e-9
