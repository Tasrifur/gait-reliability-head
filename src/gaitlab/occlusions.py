"""Deterministic GEI corruptions for robustness evaluation (Phase 7 prep).

Every function is a pure function of (image, cell, salt): the same cell gets the
same corruption for every model. The reliability head is trained ONLY with
horizontal bands (augment.erase_band); vband / bottom / noise test whether its
uncertainty generalises to unseen corruption types.

  hband  : zero a full-width horizontal band, height 6-16 rows (railing, carried object)
  vband  : zero a full-height vertical band, width 8-16 px, placed over the body
           (pole, passer-by)
  bottom : zero the lowest 14-26 rows (legs hidden by an obstacle)
  noise  : segmentation noise: ~15% of foreground pixels dropped in random 3x3
           holes + 3-6 spurious grey blobs (radius 2-5) outside the body
"""
import numpy as np

from .eval_enrollment import band_for

TYPES = ['hband', 'vband', 'bottom', 'noise']


def _rng(cell, salt):
    return np.random.RandomState((hash(cell) + 7919 * salt) % (2 ** 31))


def corrupt(img, cell, kind, salt=2):
    """img [1,64,64] float in [0,1] (numpy). Returns a corrupted copy."""
    x = img.copy()
    if kind == 'hband':
        t, h = band_for(cell, salt, 1.0)               # identical to the earlier S4 bands
        x[..., t:t + h, :] = 0
        return x
    r = _rng(cell, salt)
    if kind == 'vband':
        cols = np.where(x[0].max(0) > 0.2)[0]
        lo, hi = (cols.min(), cols.max()) if len(cols) else (16, 48)
        w = r.randint(8, 17)
        l = int(np.clip(r.randint(lo, max(lo + 1, hi - w + 1)), 0, 64 - w))
        x[..., :, l:l + w] = 0
    elif kind == 'bottom':
        h = r.randint(14, 27)
        x[..., 64 - h:, :] = 0
    elif kind == 'noise':
        fg = np.argwhere(x[0] > 0.2)
        if len(fg):
            n = int(0.15 * len(fg) / 9)
            for (i, j) in fg[r.choice(len(fg), max(n, 1), replace=False)]:
                x[0, max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = 0
        yy, xx = np.mgrid[:64, :64]
        for _ in range(r.randint(3, 7)):
            for _try in range(20):
                cy, cx, rad = r.randint(0, 64), r.randint(0, 64), r.randint(2, 6)
                if x[0, cy, cx] < 0.05:
                    break
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
            x[0][m] = np.maximum(x[0][m], r.uniform(0.3, 0.8))
    else:
        raise ValueError(kind)
    return x
