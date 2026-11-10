"""Build the CASIA-B GEI cache directly from casia_b_silhouettes.zip.

Output: data/cache/casiab_gei.npz with explicitly labelled axes
    gei     uint8  [124 subjects, 10 seqs, 11 views, 64, 64]   (GEI * 255)
    nframes int16  [124, 10, 11]   frames used (0 = missing sequence)
    nraw    int16  [124, 10, 11]   frames present before filtering
    subjects, seqs, views          label arrays for every axis

Seq axis order is fixed: nm-01..nm-06, bg-01, bg-02, cl-01, cl-02.
The input frames are already cropped/height-normalised 64x64 silhouettes, so the
GEI is the plain mean of foreground-valid frames.

Usage:  python -m gaitlab.data.build_casiab --zip <path> --out <path>
"""
import argparse
import io
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image

SEQS = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06',
        'bg-01', 'bg-02', 'cl-01', 'cl-02']
VIEWS = ['000', '018', '036', '054', '072', '090',
         '108', '126', '144', '162', '180']
MIN_FG_PIXELS = 50   # frames with fewer foreground pixels are segmentation failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    groups = defaultdict(list)
    for name in zf.namelist():
        if name.endswith('.png'):
            subj, seq, view, _ = name.split('/')
            groups[(subj, seq, view)].append(name)

    subjects = sorted({k[0] for k in groups})
    s_idx = {s: i for i, s in enumerate(subjects)}
    q_idx = {q: i for i, q in enumerate(SEQS)}
    v_idx = {v: i for i, v in enumerate(VIEWS)}

    gei = np.zeros((len(subjects), len(SEQS), len(VIEWS), 64, 64), np.uint8)
    nframes = np.zeros((len(subjects), len(SEQS), len(VIEWS)), np.int16)
    nraw = np.zeros_like(nframes)

    for n, ((subj, seq, view), names) in enumerate(sorted(groups.items())):
        frames = []
        for fn in sorted(names):
            im = np.asarray(Image.open(io.BytesIO(zf.read(fn))).convert('L'),
                            dtype=np.float32) / 255.0
            assert im.shape == (64, 64), (fn, im.shape)
            if (im > 0.5).sum() >= MIN_FG_PIXELS:
                frames.append(im)
        i, q, v = s_idx[subj], q_idx[seq], v_idx[view]
        nraw[i, q, v] = len(names)
        nframes[i, q, v] = len(frames)
        if frames:
            gei[i, q, v] = np.round(np.mean(frames, axis=0) * 255).astype(np.uint8)
        if n % 1000 == 0:
            print(f'{n}/{len(groups)} sequences', flush=True)

    np.savez_compressed(args.out, gei=gei, nframes=nframes, nraw=nraw,
                        subjects=np.array(subjects), seqs=np.array(SEQS),
                        views=np.array(VIEWS))
    print('saved', args.out, gei.shape,
          'missing:', int((nframes == 0).sum()),
          'short(<15):', int(((nframes > 0) & (nframes < 15)).sum()))


if __name__ == '__main__':
    main()
