"""Build GEI caches for CASIA-E and OU-MVLP directly from their zips.

Inputs are 88x128 (W x H) GEIs. Each is zero-padded symmetrically to 128x128
(preserving aspect ratio) and resized to 64x64 with area averaging, matching
the CASIA-B cache geometry.

Output (data/cache/<name>_gei.npy + <name>_meta.csv):
    <name>_gei.npy   uint8 [N, 64, 64]
    <name>_meta.csv  one row per GEI, row i <-> gei[i]
        casiae: subject,height,scene,variant,cond,angle,seq,file
        oumvlp: subject,angle,seq,file

CASIA-E note: at angles 045 and 135 there are seqs 1-4 (other angles 1-2).
Seqs 3/4 look like a different camera/walk direction, so `seq` is kept
verbatim and samplers must decide how to treat it. Nothing is merged here.

Usage: python -m gaitlab.data.build_gei_zip --dataset casiae --zip ... --outdir ...
       (or --dir <unzipped root> instead of --zip)
"""
import argparse
import csv
import io
import os
import re
import zipfile

import numpy as np
from PIL import Image

CASIAE_RE = re.compile(
    r'^(\d{4})/(\d{4})_([HL])_(scene[123])_(?:(s)_)?(nm|bg|cl)_(\d{3})_(\d+)\.png$')
OUMVLP_RE = re.compile(r'^(\d{3})-(\d{2})/(\d{5})\.png$')


def to64(png_bytes):
    im = Image.open(io.BytesIO(png_bytes)).convert('L')
    w, h = im.size
    side = max(w, h)
    canvas = Image.new('L', (side, side), 0)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return np.asarray(canvas.resize((64, 64), Image.BOX), dtype=np.uint8)


def parse(dataset, name):
    if dataset == 'casiae':
        m = CASIAE_RE.match(name)
        if not m:
            return None
        d, subj, height, scene, var, cond, angle, seq = m.groups()
        assert d == subj, name
        return dict(subject=subj, height=height, scene=scene,
                    variant=var or '', cond=cond, angle=angle, seq=int(seq),
                    file=name)
    m = OUMVLP_RE.match(name)
    if not m:
        return None
    angle, seq, subj = m.groups()
    return dict(subject=subj, angle=angle, seq=int(seq), file=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['casiae', 'oumvlp'], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--zip')
    src.add_argument('--dir', help='already-unzipped root; same relative layout as the zip')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    if args.zip:
        zf = zipfile.ZipFile(args.zip)
        names, read = zf.namelist(), zf.read
    else:
        names = [os.path.relpath(os.path.join(d, f), args.dir).replace(os.sep, '/')
                 for d, _, fs in os.walk(args.dir) for f in fs]
        read = lambda n: open(os.path.join(args.dir, n), 'rb').read()
    rows, skipped = [], []
    for name in sorted(names):
        if not name.endswith('.png'):
            continue
        r = parse(args.dataset, name)
        (rows if r else skipped).append(r or name)
    print(f'{len(rows)} GEIs parsed, {len(skipped)} unparsed', skipped[:5], flush=True)

    out_npy = os.path.join(args.outdir, f'{args.dataset}_gei.npy')
    arr = np.lib.format.open_memmap(out_npy, mode='w+', dtype=np.uint8,
                                    shape=(len(rows), 64, 64))
    for i, r in enumerate(rows):
        arr[i] = to64(read(r['file']))
        if i % 50000 == 0:
            print(f'{i}/{len(rows)}', flush=True)
    arr.flush()

    with open(os.path.join(args.outdir, f'{args.dataset}_meta.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('saved', out_npy, arr.shape)


if __name__ == '__main__':
    main()
