"""Calibrate the reliability head on HELD-OUT identities (never used to train the backbone).

Same objective as Phase 4 (two-stage, frozen backbone, occlusion-aware positive-pair
Gaussian NLL, horizontal band erasing p=0.5), but the episodes are drawn ONLY from
the held-out calibration identities of each dataset:
    CASIA-B : the 12 LT validation subjects (001-074 minus B1's 62 training subjects)
    CASIA-E : 40 held-out ids of the official training set (001-200)
    OU-MVLP : 300 held-out ids of the official training set (odd ids)

Usage: python -m gaitlab.calibrate --dataset casiab|casiae|oumvlp --b1 <run dir> --out <dir>
"""
import argparse
import json
import os

import numpy as np
import torch

from .augment import erase_band
from .data.casiab import CasiaB, COND_SEQS, SEQS, VIEWS
from .data.dense import DenseGEI
from .data.sampler import EpisodeSampler
from .models.probnet import ProbProtoNet
from .utils import save_json, set_seed

N_CAL = {'casiae': 40, 'oumvlp': 300}


def load_dataset(name):
    if name == 'casiab':
        ds = CasiaB('data/cache/casiab_gei.npz')
        sp = ds.lt_split(12)
        return ds, sp, COND_SEQS['nm'], SEQS, VIEWS
    ds = DenseGEI(name)
    return ds, ds.split(N_CAL[name]), ds.seq_set('nm_all'), ds.seq_set('all'), ds.views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--b1', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--erase_p', type=float, default=0.5)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    ds, sp, gseqs, pseqs, views = load_dataset(a.dataset)
    cal = sp['val']
    n_way = min(20, len(cal) - 2)
    for seed in a.seeds:
        set_seed(seed)
        pm = ProbProtoNet('v0bn', logdet_weight=1.0, var_init=4.0).to(a.device)
        b1 = torch.load(os.path.join(a.b1, f'seed{seed}', 'best.pt'), weights_only=True)
        pm.cnn.load_state_dict({k[4:]: v for k, v in b1.items() if k.startswith('cnn.')})
        for p in pm.cnn.parameters():
            p.requires_grad_(False)
        pm.cnn.eval()
        opt = torch.optim.Adam(list(pm.var_head.parameters()) + [pm.part_bias], lr=1e-3)
        smp = EpisodeSampler(ds, cal, gseqs, pseqs, views, views, n_way, 4, 5,
                             same_view=False, rng=np.random.RandomState(seed))
        for it in range(1, a.iters + 1):
            pm.var_head.train()
            sup, qry, lab, _ = smp.sample()
            S, Q = smp.to_tensors(sup).to(a.device), smp.to_tensors(qry).to(a.device)
            S, Q = erase_band(S, a.erase_p), erase_band(Q, a.erase_p)
            pm(S, Q, n_way, 4)
            loss = pm.pos_nll_detached_means(torch.from_numpy(lab).to(a.device)).mean() / pm.D
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 1000 == 0:
                print(f'{a.dataset} seed {seed} it {it} pos-NLL/D {loss.item():.4f}', flush=True)
        os.makedirs(os.path.join(a.out, f'seed{seed}'), exist_ok=True)
        torch.save({k: v for k, v in pm.state_dict().items() if not k.startswith('cnn.')},
                   os.path.join(a.out, f'seed{seed}', 'var_head.pt'))
        save_json({'dataset': a.dataset, 'b1': a.b1, 'calibration_ids': len(cal), 'n_way': n_way,
                   'iters': a.iters, 'erase_p': a.erase_p}, os.path.join(a.out, f'seed{seed}', 'calib.json'))


if __name__ == '__main__':
    main()
