"""Episodic training + evaluation on CASIA-B.

Usage:
    python -m gaitlab.train --config configs/v0_dppnet.yaml [--seeds 0 1 2 3 4]

Writes results/<name>/seed<k>/{config.json, metrics.json, best.pt}.
Every number later reported must come from these metrics.json files.
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from . import evaluate
from .data.casiab import CasiaB, COND_SEQS, SEQS, VIEWS
from .data.dense import DenseGEI
from .data.sampler import EpisodeSampler, LegacySampler
from .models.dppnet import DPPNet
from .models.protonet import ProtoNet
from .models.probnet import ProbProtoNet
from .models.heads import HeadNet
from .augment import augment
from .utils import provenance, save_json, set_seed

SEQ_SETS = {'nm_all': COND_SEQS['nm'], 'all': SEQS,
            'lt_gallery': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
            'legacy_gallery': ['nm-05', 'nm-06']}


def build_model(cfg):
    m = cfg['model']
    if m['name'] == 'dppnet':
        return DPPNet(m.get('hidden_dim', 128), m.get('num_parts', 8),
                      m.get('attention', 'residual'))
    if m['name'] == 'protonet':
        return ProtoNet(m.get('encoder', 'v0'), m.get('width', 1.0), m.get('scales', [8]),
                        m.get('pool', 'mean'), m.get('l2norm', False),
                        m.get('scale', 1.0), m.get('scale_init', 10.0))
    if m['name'] == 'head':
        return HeadNet(m['head'])
    if m['name'] == 'probnet':
        return ProbProtoNet(m.get('encoder', 'v0bn'), m.get('width', 1.0), m.get('num_parts', 8),
                            m.get('fusion', 'precision'), m.get('logdet_weight', 1.0),
                            m.get('var_init', 4.0), m.get('var_floor', 1e-3))
    raise ValueError(m['name'])


def pretrain_classifier(model, ds, subjects, steps, lr, device, seed, batch=128):
    """Meta-Baseline stage 1: standard CE classification over the training
    identities (all their valid GEIs), linear classifier on the embedding."""
    rng = np.random.RandomState(seed)
    cells = [(s, q, v) for s in subjects for q in range(10) for v in range(11) if ds.valid[s, q, v]]
    lab = {s: i for i, s in enumerate(subjects)}
    clf = nn.Linear(model.D, len(subjects)).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=lr, weight_decay=1e-5)
    for it in range(1, steps + 1):
        model.train()
        b = [cells[i] for i in rng.choice(len(cells), batch, replace=False)]
        X = torch.from_numpy(np.stack([ds.get(*c) for c in b])).to(device)
        y = torch.tensor([lab[c[0]] for c in b], device=device)
        loss = nn.functional.cross_entropy(clf(model.embed(X)), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f'  pretrain step {it} loss {loss.item():.3f}', flush=True)


def run_seed(cfg, seed, ds, split, out_dir, device):
    set_seed(seed)
    model = build_model(cfg).to(device)
    t = cfg['train']
    opt = torch.optim.Adam(model.parameters(), lr=t['lr'], weight_decay=t['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t['max_episodes'], eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()
    Sampler = LegacySampler if t.get('sampler') == 'legacy' else EpisodeSampler
    dense = isinstance(ds, DenseGEI)
    seqset = ds.seq_set if dense else SEQ_SETS.__getitem__
    views = ds.views if dense else VIEWS
    smp = Sampler(ds, split['train'], seqset(t['gallery_seqs']),
                         seqset(t['probe_seqs']), views, views,
                         t['n_way'], t['k_shot'], t['n_query'],
                         same_view=t['same_view'], rng=np.random.RandomState(seed))
    if t.get('pretrain_steps'):
        pretrain_classifier(model, ds, split['train'], t['pretrain_steps'], t['lr'], device, seed)
    v = cfg['val']
    best, bad, best_ep, hist = -1.0, 0, 0, []
    ckpt = os.path.join(out_dir, 'best.pt')
    t0 = time.time()
    for ep in range(1, t['max_episodes'] + 1):
        model.train()
        sup, qry, lab, _ = smp.sample()
        S, Q = smp.to_tensors(sup).to(device), smp.to_tensors(qry).to(device)
        aug = t.get('augment')
        if aug:
            S, Q = augment(S, **aug), augment(Q, **aug)
        y = torch.from_numpy(lab).to(device)
        aux_w = t.get('aux_part_loss', 0.0)
        mls_w = t.get('mls_loss', 0.0)
        if aux_w or mls_w:
            logits, _, part_logits = model(S, Q, t['n_way'], t['k_shot'], return_part_logits=True)
            loss = loss_fn(logits, y)
            if aux_w:
                aux = torch.stack([loss_fn(part_logits[..., i], y)
                                   for i in range(part_logits.size(-1))]).mean()
                loss = loss + aux_w * aux
            if mls_w:
                # positive-pair Gaussian NLL (PFE-style mutual likelihood): trains the
                # variance to match each strip's actual query-to-own-prototype discrepancy
                pos = model.pos_nll_detached_means(y)                          # [Nq, P]
                loss = loss + mls_w * pos.mean() / model.D
        else:
            logits, _ = model(S, Q, t['n_way'], t['k_shot'])
            loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % v['every'] == 0:
            if dense:   # one validation pool: NM support, any-condition queries, held-out ids
                accs = [evaluate.episodic(model, ds, split['val'], None, v['n_way'], v['k_shot'],
                                          v['n_query'], v['same_view'], v['episodes'], seed=777,
                                          device=device, gallery_seqs=ds.seq_set('nm_all'),
                                          probe_seqs=ds.seq_set('all'), views=ds.views)['mean']]
            else:
                accs = [evaluate.episodic(model, ds, split['val'], c, v['n_way'], v['k_shot'],
                                          v['n_query'], v['same_view'], v['episodes'],
                                          seed=777, device=device,
                                          legacy=v.get('legacy', False))['mean']
                        for c in v['conds']]
            score = float(np.mean(accs))
            hist.append({'episode': ep, 'loss': float(loss), 'val': score})
            print(f'  seed {seed} ep {ep} loss {loss.item():.3f} val {score:.2f}', flush=True)
            if score > best:
                best, bad, best_ep = score, 0, ep
                torch.save(model.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= v['patience']:
                    break
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    train_time = time.time() - t0

    results = {}
    for e in cfg['eval']:
        key = e['name']
        if e['kind'] == 'episodic':
            results[key] = {c: evaluate.episodic(model, ds, split['test'], c, e['n_way'],
                                                 e['k_shot'], e['n_query'], e['same_view'],
                                                 e['episodes'], seed=12345, device=device,
                                                 legacy=e.get('legacy', False))
                            for c in ['nm', 'bg', 'cl']}
        elif e['kind'] == 'full_gallery':
            results[key] = {c: evaluate.full_gallery_rank1(model, ds, split['test'], c,
                                                           e['cross_view'], device)
                            for c in ['nm', 'bg', 'cl']}
        summ = {c: results[key][c]['mean'] for c in ['nm', 'bg', 'cl']}
        print(f'  [{key}] ' + ' '.join(f'{c.upper()} {a:.2f}' for c, a in summ.items())
              + f' | mean {np.mean(list(summ.values())):.2f}', flush=True)

    save_json({'seed': seed, 'best_val': best, 'best_episode': best_ep,
               'train_time_s': train_time, 'history': hist, 'results': results,
               'n_params': sum(p.numel() for p in model.parameters()),
               'provenance': provenance()},
              os.path.join(out_dir, 'metrics.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', type=int, nargs='+')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seeds = args.seeds or cfg.get('seeds', [0])
    if cfg['data'].get('type') == 'dense':
        ds = DenseGEI(cfg['data']['name'])
        split = ds.split(cfg['data']['n_cal'])
    else:
        ds = CasiaB(cfg['data']['cache'], cfg['data'].get('min_frames', 15))
        split = ds.lt_split(cfg['data'].get('n_val', 10)) if cfg['data']['split'] == 'lt' \
            else ds.legacy_split()
    for seed in seeds:
        out_dir = os.path.join('results', cfg['name'], f'seed{seed}')
        if os.path.exists(os.path.join(out_dir, 'metrics.json')):
            print(f'=== {cfg["name"]} seed {seed}: metrics.json exists, skipping', flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        save_json(copy.deepcopy(cfg), os.path.join(out_dir, 'config.json'))
        print(f'=== {cfg["name"]} seed {seed}', flush=True)
        run_seed(cfg, seed, ds, split, out_dir, args.device)


if __name__ == '__main__':
    main()
