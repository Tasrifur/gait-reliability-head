"""Write CASIA-B silhouettes in OpenGait's pickle layout + a custom partition.

Frames in casia_b_silhouettes.zip are already OpenGait-style preprocessed 64x64
(top/bottom crop, height 64, centred on the median column), so each sequence
is just stacked: data/opengait_pkl/<sid>/<seq>/<view>/<view>.pkl (uint8 [T,64,64]).
Frames with <50 foreground pixels are dropped and only cells valid in the GEI
cache (>=15 frames) are written, so every pipeline sees the same sequences.

Partition (data/opengait_pkl/CASIA-B_062.json):
  TRAIN_SET 001-062   (backbone training)
  TEST_SET  075-124   (never used for training)
  063-074 are deliberately in neither set: they are the held-out
  calibration subjects for the reliability head.

Usage: python -m gaitlab.data.build_opengait_pkl
"""
import io
import json
import os
import pickle
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image

from .casiab import CasiaB, SEQS, VIEWS

OUT = 'data/opengait_pkl'


def main():
    ds = CasiaB('data/cache/casiab_gei.npz')
    zf = zipfile.ZipFile('casia_b_silhouettes.zip')
    groups = defaultdict(list)
    for n in zf.namelist():
        if n.endswith('.png'):
            s, q, v, _ = n.split('/')
            groups[(s, q, v)].append(n)
    written = 0
    for (s, q, v), files in sorted(groups.items()):
        if not ds.valid[ds.sidx(s), SEQS.index(q), VIEWS.index(v)]:
            continue
        fr = np.stack([np.asarray(Image.open(io.BytesIO(zf.read(f))).convert('L'), np.uint8)
                       for f in sorted(files)])
        fr = fr[(fr > 127).sum((1, 2)) >= 50]
        d = os.path.join(OUT, s, q, v)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f'{v}.pkl'), 'wb') as f:
            pickle.dump(fr, f)
        written += 1
    part = {'TRAIN_SET': [f'{i:03d}' for i in range(1, 63)],
            'TEST_SET': [f'{i:03d}' for i in range(75, 125)]}
    json.dump(part, open(os.path.join(OUT, 'CASIA-B_062.json'), 'w'), indent=1)
    print('sequences written', written)


if __name__ == '__main__':
    main()
