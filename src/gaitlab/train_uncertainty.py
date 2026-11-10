"""Phase 4, two-stage N1/N2: learn per-strip uncertainty on a frozen B1 encoder.

Stage 1 = B1 (results/p3_bn_aux/seed<k>/best.pt), unchanged.
Stage 2 = train only the variance head (+ per-strip bias) with the positive-pair
Gaussian NLL (PFE-style): for each query and its OWN identity's prototype,
    sum_i ||mu_q,i - mu_c,i||^2 / (s_q,i + s_c,i) + D log(s_q,i + s_c,i)
which is minimised when s_q,i + s_c,i matches that strip's expected squared
discrepancy. Means are frozen, so the scale degeneracy cannot occur.

At test time, on the SAME frozen encoder, we compare:
    b1         uniform distance (= B1)
    n1         inverse-variance distance, plain-mean prototypes
    n1n2       inverse-variance distance, precision-weighted prototypes
each with lam=1 (exact NLL incl. log term) and lam=0 (weights only).

Usage: python -m gaitlab.train_uncertainty --b1 results/p3_bn_aux --out results/p4_2stage
"""
import argparse
import copy
import glob
import json
import os

import numpy as np
import torch

from . import evaluate
from .data.casiab import CasiaB, COND_SEQS, SEQS, VIEWS
from .data.sampler import EpisodeSampler
from .augment import erase_band
from .models.probnet import ProbProtoNet
from .utils import provenance, save_json, set_seed

VARIANTS = {'b1': None,
            'n1_lam1': ('mean', 1.0), 'n1_lam0': ('mean', 0.0),
            'n1n2_lam1': ('precision', 1.0), 'n1n2_lam0': ('precision', 0.0)}


class Uniform(torch.nn.Module):
    """B1 distance on the probnet's (frozen) encoder: -mean_i ||q_i - c_i||^2."""
    def __init__(self, pm):
        super().__init__()
        self.pm = pm

    def logits_unequal(self, support, owner, n_way, query, chunk=256):
        S = torch.cat([self.pm.encoder(support[i:i + chunk]) for i in range(0, len(support), chunk)])
        c = torch.zeros(n_way, self.pm.P, self.pm.D, device=S.device).index_add_(0, owner, S)
        c /= torch.bincount(owner, minlength=n_way).clamp(min=1)[:, None, None].float()
        return torch.cat([-((self.pm.encoder(query[i:i + chunk])[:, None] - c[None]) ** 2).sum(-1).mean(-1)
                          for i in range(0, len(query), chunk)])

    def forward(self, support, query, n_way, k_shot):
        owner = torch.arange(n_way, device=support.device).repeat_interleave(k_shot)
        return self.logits_unequal(support, owner, n_way, query), None


def make_variant(pm, name):
    if VARIANTS[name] is None:
        return Uniform(pm)
    v = copy.copy(pm)                       # shallow: shares weights
    v.fusion, v.lam = VARIANTS[name]
    return v


def run_seed(b1_dir, seed, out_dir, a, ds, split, device):
    set_seed(seed)
    cfg = json.load(open(os.path.join(b1_dir, 'config.json')))
    pm = ProbProtoNet(cfg['model'].get('encoder', 'v0bn'), fusion=a.train_fusion,
                      logdet_weight=1.0, var_init=a.var_init).to(device)
    b1 = torch.load(os.path.join(b1_dir, 'best.pt'), weights_only=True)
    pm.cnn.load_state_dict({k[4:]: v for k, v in b1.items() if k.startswith('cnn.')})
    for p in pm.cnn.parameters():
        p.requires_grad_(False)
    pm.cnn.eval()
    params = list(pm.var_head.parameters()) + [pm.part_bias]
    opt = torch.optim.Adam(params, lr=a.lr)
    t = cfg['train']
    smp = EpisodeSampler(ds, split['train'], COND_SEQS['nm'], SEQS, VIEWS, VIEWS,
                         t['n_way'], t['k_shot'], t['n_query'], same_view=False,
                         rng=np.random.RandomState(seed))
    for it in range(1, a.iters + 1):
        pm.var_head.train()
        sup, qry, lab, _ = smp.sample()
        S, Q = smp.to_tensors(sup).to(device), smp.to_tensors(qry).to(device)
        if a.erase_p > 0:
            S, Q = erase_band(S, a.erase_p), erase_band(Q, a.erase_p)
        pm(S, Q, t['n_way'], t['k_shot'])
        loss = pm.pos_nll_detached_means(torch.from_numpy(lab).to(device)).mean() / pm.D
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0:
            print(f'  seed {seed} it {it} pos-NLL/D {loss.item():.4f}', flush=True)
    torch.save({k: v for k, v in pm.state_dict().items() if not k.startswith('cnn.')},
               os.path.join(out_dir, 'var_head.pt'))
    pm.eval()

    results = {}
    for name in VARIANTS:
        model = make_variant(pm, name)
        results[f'fgcv_{name}'] = {c: evaluate.full_gallery_rank1(model, ds, split['test'], c, True, device)
                                   for c in ['nm', 'bg', 'cl']}
        results[f'epcv_{name}'] = {c: evaluate.episodic(model, ds, split['test'], c, 20, 4, 2, False,
                                                        500, seed=12345, device=device)
                                   for c in ['nm', 'bg', 'cl']}
        s = {c: results[f'fgcv_{name}'][c]['mean'] for c in ['nm', 'bg', 'cl']}
        print(f'  [{name}] FG-CV ' + ' '.join(f'{c.upper()} {x:.2f}' for c, x in s.items())
              + f' | mean {np.mean(list(s.values())):.2f}', flush=True)
    save_json({'seed': seed, 'best_val': None, 'best_episode': a.iters, 'results': results,
               'n_params': sum(p.numel() for p in pm.parameters()), 'args': vars(a),
               'provenance': provenance()}, os.path.join(out_dir, 'metrics.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b1', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--var_init', type=float, default=4.0)
    ap.add_argument('--train_fusion', default='precision')
    ap.add_argument('--erase_p', type=float, default=0.0,
                    help='prob. of blanking a random horizontal band in each support/query image')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    ds = CasiaB('data/cache/casiab_gei.npz')
    for d in sorted(glob.glob(os.path.join(a.b1, 'seed*'))):
        seed = int(os.path.basename(d)[4:])
        cfg = json.load(open(os.path.join(d, 'config.json')))
        split = ds.lt_split(cfg['data'].get('n_val', 10))
        out = os.path.join(a.out, f'seed{seed}')
        os.makedirs(out, exist_ok=True)
        print(f'=== two-stage uncertainty on {d}', flush=True)
        run_seed(d, seed, out, a, ds, split, a.device)


if __name__ == '__main__':
    main()
