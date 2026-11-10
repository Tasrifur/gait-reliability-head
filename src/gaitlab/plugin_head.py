"""Phase 5c: the N1/N2/N4 reliability head as a PLUG-IN on frozen GaitBase/GaitSet.

The frozen backbone gives per-part embeddings mu [P, C] (GaitBase P=16, GaitSet
P=62, C=256). A small variance head (shared MLP over parts + per-part bias)
predicts s_p > 0 for every part of every sequence. It is trained exactly as in
Phase 4 (two-stage, positive-pair Gaussian NLL against the subject's own
K-shot prototype, occlusion-aware: support/query sequences are replaced by a
band-occluded version with p=0.5). The backbone is never updated.

Stages (cached, re-runnable):
  embed : frozen embeddings of all TRAINING-subject sequences (001-074), clean
          and two randomly band-occluded versions (bands differ from the test salts)
  train : 5 seeds of the variance head per backbone
  eval  : on the cached TEST embeddings (results/p5_frozen/embeddings.pt):
          FG-CV, S1/S3/S4 scenarios and open-set, for
            frozen    : backbone's own metric (mean per-part Euclidean), prototypes
            sq        : uniform squared Euclidean, prototypes (metric control)
            n1n2      : Gaussian NLL with learned variances + precision fusion
          open-set scores: raw/margin (frozen, sq) and likelihood ratio (n1n2)

Usage: python -m gaitlab.plugin_head embed|train|eval
"""
import io
import json
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .data.casiab import CasiaB, COND_SEQS, LT_GALLERY, LT_PROBES, SEQS, VIEWS
from .eval_enrollment import GAL, PRB
from .eval_openset import FARS, auroc, dir_at_far, oscr
from .external_eval import build, dist
from .utils import mean_ci, save_json

OUT = 'results/p5_plugin'
BACKBONES = ['gaitbase', 'gaitset']


# ------------------------------------------------------------------ embed ----
@torch.no_grad()
def embed_train(device='cuda'):
    ds = CasiaB('data/cache/casiab_gei.npz')
    train = [i for i, s in enumerate(ds.subjects) if int(s) <= 74]
    models = {n: build(n).to(device) for n in BACKBONES}
    zf = zipfile.ZipFile('casia_b_silhouettes.zip')
    groups = defaultdict(list)
    for n in zf.namelist():
        if n.endswith('.png'):
            s, q, v, _ = n.split('/')
            groups[(s, q, v)].append(n)
    cells = [(s, q, v) for s in train for q in range(10) for v in range(11) if ds.valid[s, q, v]]
    rng = np.random.RandomState(777)                     # bands independent of the test salts
    out = {n: {k: [] for k in ['clean', 'occA', 'occB']} for n in models}
    for i, (s, q, v) in enumerate(cells):
        files = sorted(groups[(ds.subjects[s], SEQS[q], VIEWS[v])])
        fr = np.stack([np.asarray(Image.open(io.BytesIO(zf.read(f))).convert('L'), np.float32)
                       for f in files])
        fr = fr[(fr > 127).sum((1, 2)) >= 50][:720]
        var = {'clean': fr}
        for k in ['occA', 'occB']:
            h = rng.randint(6, 17); t = rng.randint(0, 64 - h + 1)
            f2 = fr.copy(); f2[:, t:t + h, :] = 0
            var[k] = f2
        for k, x in var.items():
            ten = torch.from_numpy(x[:, :, 10:-10] / 255.0)[None, None].to(device)
            for n, m in models.items():
                out[n][k].append(m(ten)[0].cpu())
        if i % 1000 == 0:
            print(f'{i}/{len(cells)} training sequences embedded', flush=True)
    os.makedirs(OUT, exist_ok=True)
    for n in models:
        torch.save((cells, {k: torch.stack(v) for k, v in out[n].items()}), f'{OUT}/train_emb_{n}.pt')


# ------------------------------------------------------------------- head ----
class VarHead(nn.Module):
    def __init__(self, P, C=256, var_init=1.0, floor=1e-3):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(C, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        nn.init.zeros_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias)
        self.bias = nn.Parameter(torch.full((P,), float(np.log(np.expm1(var_init)))))
        self.floor = floor

    def forward(self, E):                                  # E [n, C, P] -> var [n, P]
        return F.softplus(self.mlp(E.transpose(1, 2)).squeeze(-1) + self.bias) + self.floor


def fuse(mu, var, owner, n):
    """Precision-weighted prototypes. mu [m,C,P], var [m,P]."""
    prec = 1.0 / var
    ps = torch.zeros(n, var.shape[1], device=mu.device).index_add_(0, owner, prec)
    ms = torch.zeros(n, *mu.shape[1:], device=mu.device).index_add_(0, owner, mu * prec[:, None])
    return ms / ps[:, None], 1.0 / ps


def nll_logits(mq, vq, mc, vc):
    """-0.5 sum_p [||mq-mc||^2/(vq+vc) + C log(vq+vc)].  mq [q,C,P] mc [n,C,P]."""
    s = vq[:, None] + vc[None]                              # [q,n,P]
    sq = ((mq[:, None] - mc[None]) ** 2).sum(2)             # [q,n,P]
    return -0.5 * (sq / s + mq.shape[1] * torch.log(s)).sum(-1)


def train_heads(device='cuda', iters=3000, n_way=20, k_shot=4, n_query=5, erase_p=0.5):
    ds = CasiaB('data/cache/casiab_gei.npz')
    for bb in BACKBONES:
        cells, E = torch.load(f'{OUT}/train_emb_{bb}.pt', weights_only=False)
        E = {k: v.to(device) for k, v in E.items()}
        idx = {c: i for i, c in enumerate(cells)}
        subjects = sorted({c[0] for c in cells})
        nm = [ds.qidx(q) for q in COND_SEQS['nm']]
        P = E['clean'].shape[2]
        for seed in range(5):
            torch.manual_seed(seed); rng = np.random.RandomState(seed)
            head = VarHead(P).to(device)
            opt = torch.optim.Adam(head.parameters(), lr=1e-3)
            for it in range(1, iters + 1):
                ids = rng.choice(subjects, n_way, replace=False)
                sup, qry, lab = [], [], []
                for c, s in enumerate(ids):
                    g = [(s, q, v) for q in nm for v in range(11) if (s, q, v) in idx]
                    sc = [g[i] for i in rng.choice(len(g), k_shot, replace=False)]
                    used = {x[2] for x in sc}
                    p = [(s, q, v) for q in range(10) for v in range(11)
                         if (s, q, v) in idx and v not in used and (s, q, v) not in sc]
                    qc = [p[i] for i in rng.choice(len(p), min(n_query, len(p)), replace=False)]
                    sup += sc; qry += qc; lab += [c] * len(qc)

                def pick(cs):
                    keys = ['clean' if rng.rand() >= erase_p else ('occA' if rng.rand() < .5 else 'occB')
                            for _ in cs]
                    return torch.stack([E[k][idx[c]] for k, c in zip(keys, cs)])
                Ms, Mq = pick(sup), pick(qry)
                vs, vq = head(Ms), head(Mq)
                owner = torch.arange(n_way, device=device).repeat_interleave(k_shot)
                mc, vc = fuse(Ms, vs, owner, n_way)
                y = torch.tensor(lab, device=device)
                s = vq + vc[y]
                sqd = ((Mq - mc[y]) ** 2).sum(1)
                loss = (0.5 * (sqd / s + Mq.shape[1] * torch.log(s))).mean() / Mq.shape[1]
                opt.zero_grad(); loss.backward(); opt.step()
                if it % 1000 == 0:
                    print(f'{bb} seed {seed} it {it} pos-NLL/C {loss.item():.4f}', flush=True)
            torch.save(head.state_dict(), f'{OUT}/head_{bb}_seed{seed}.pt')


# ------------------------------------------------------------------- eval ----
@torch.no_grad()
def evaluate_backbone(bb, cells, E, head, ds, test, device, kinds=None, closed=True):
    idx = {c: i for i, c in enumerate(cells)}
    M = {k: v.to(device) for k, v in E.items()}
    V = {k: head(v) for k, v in M.items()} if head is not None else None
    gq = [ds.qidx(x) for x in LT_GALLERY]

    def logits(kind, gcells_by_id, gkey, pcells, pkey):
        if kind == 'frozen':
            proto = torch.stack([M[gkey][[idx[c] for c in cs]].mean(0) for cs in gcells_by_id])
            return -dist(M[pkey][[idx[c] for c in pcells]], proto)
        if kind == 'sq':
            proto = torch.stack([M[gkey][[idx[c] for c in cs]].mean(0) for cs in gcells_by_id])
            q = M[pkey][[idx[c] for c in pcells]]
            return -((q[:, None] - proto[None]) ** 2).sum((2, 3))
        rows = [idx[c] for cs in gcells_by_id for c in cs]
        owner = torch.tensor([j for j, cs in enumerate(gcells_by_id) for _ in cs], device=device)
        mc, vc = fuse(M[gkey][rows], V[gkey][rows], owner, len(gcells_by_id))
        pr = [idx[c] for c in pcells]
        return nll_logits(M[pkey][pr], V[pkey][pr], mc, vc), (mc, vc, M[pkey][pr], V[pkey][pr])

    kinds = kinds or (['frozen', 'sq'] + (['n1n2'] if head is not None else []))
    res = {k: {} for k in kinds}

    def fg(kind, gseqs, pseqs, gkey='clean', pkey='clean'):
        gq_, pq_ = [ds.qidx(x) for x in gseqs], [ds.qidx(x) for x in pseqs]
        accs = []
        for pv in range(11):
            per = []
            for gv in range(11):
                if gv == pv:
                    continue
                ids = [s for s in test if any((s, x, gv) in idx for x in gq_)]
                gc = [[(s, x, gv) for x in gq_ if (s, x, gv) in idx] for s in ids]
                p = [(s, x, pv) for s in ids for x in pq_ if (s, x, pv) in idx]
                if not p:
                    continue
                lg = logits(kind, gc, gkey, p, pkey)
                lg = lg[0] if isinstance(lg, tuple) else lg
                pred = lg.argmax(1).cpu().numpy()
                per.append((np.array([ids[j] for j in pred]) == np.array([c[0] for c in p])).mean() * 100)
            accs.append(np.mean(per))
        return float(np.mean(accs))

    for kind in kinds:
        r = res[kind]
        for cond in (['nm', 'bg', 'cl'] if closed else []):
            r[f'fgcv>{cond}'] = fg(kind, LT_GALLERY, LT_PROBES[cond])
            r[f'S3_corrupt>{cond}'] = fg(kind, LT_GALLERY, LT_PROBES[cond], gkey='occ1')
            r[f'S4_occprobe>{cond}'] = fg(kind, LT_GALLERY, LT_PROBES[cond], pkey='occ2')
        for g in (['nm', 'bg', 'cl'] if closed else []):
            for p in ['nm', 'bg', 'cl']:
                if g != p:
                    r[f'S1_{g}>{p}'] = fg(kind, GAL[g], PRB[p])
        # open-set (same splits as eval_openset)
        for mode in ['nm', 'bg', 'cl', 'nm_occ']:
            cond = 'nm' if mode == 'nm_occ' else mode
            pkey = 'occ2' if mode == 'nm_occ' else 'clean'
            pq = [ds.qidx(x) for x in LT_PROBES[cond]]
            rng = np.random.RandomState(2024)
            reps = []
            for _ in range(10):
                enr = set(sorted(rng.choice(test, len(test) // 2, replace=False).tolist()))
                rules = ['raw', 'margin'] + (['lr'] if kind == 'n1n2' else [])
                sk, su, ck = {x: [] for x in rules}, {x: [] for x in rules}, []
                for gv in range(11):
                    ids = [s for s in sorted(enr) if any((s, x, gv) in idx for x in gq)]
                    gc = [[(s, x, gv) for x in gq if (s, x, gv) in idx] for s in ids]
                    for pv in range(11):
                        if pv == gv:
                            continue
                        p = [(s, x, pv) for s in test for x in pq if (s, x, pv) in idx]
                        out = logits(kind, gc, 'clean', p, pkey)
                        lg = out[0] if isinstance(out, tuple) else out
                        t2 = lg.topk(2, 1)
                        sc = {'raw': t2.values[:, 0], 'margin': t2.values[:, 0] - t2.values[:, 1]}
                        if kind == 'n1n2':
                            mc, vc, mq, vq = out[1]
                            mpop = mc.mean(0, keepdim=True)
                            vpop = (vc.mean(0) + ((mc - mpop) ** 2).sum(1).mean(0) / mc.shape[1])[None]
                            sc['lr'] = t2.values[:, 0] - nll_logits(mq, vq, mpop, vpop)[:, 0]
                        known = np.array([c[0] in enr for c in p])
                        am = t2.indices[:, 0].cpu().numpy()
                        ck += [ids[a] == c[0] for c, a, k in zip(p, am, known) if k]
                        for x in rules:
                            v = sc[x].cpu().numpy()
                            sk[x].append(v[known]); su[x].append(v[~known])
                ck = np.array(ck)
                o = {}
                for x in rules:
                    k_, u_ = np.concatenate(sk[x]), np.concatenate(su[x])
                    o[f'{x}/auroc'], o[f'{x}/oscr'] = auroc(k_, u_), oscr(k_, ck, u_)
                    for f in FARS:
                        o[f'{x}/dir@far{int(f * 100)}'] = dir_at_far(k_, ck, u_, f)
                reps.append(o)
            for k in reps[0]:
                r[f'os_{mode}/{k}'] = float(np.mean([z[k] for z in reps]))
    return res


def eval_all(device='cuda'):
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    cells, E = torch.load('results/p5_frozen/embeddings.pt', weights_only=False)
    per = {}
    for bb in BACKBONES:
        P = E[bb]['clean'].shape[2]
        per[bb] = {}
        for seed in range(5):
            head = VarHead(P).to(device).eval()
            head.load_state_dict(torch.load(f'{OUT}/head_{bb}_seed{seed}.pt', weights_only=True))
            r = evaluate_backbone(bb, cells, E[bb], head, ds, test, device,
                                  kinds=None if seed == 0 else ['n1n2'])   # frozen/sq are seed-independent
            per[bb][seed] = r
            print(bb, seed, {k: round(np.mean([r[k][f'fgcv>{c}'] for c in ['nm', 'bg', 'cl']]), 2) for k in r},
                  flush=True)
            save_json(per, f'{OUT}/eval_per_seed.json')


def fit_head(cells, E, subjects, seed, ds, device, occ_keys, iters=3000, n_way=12, k_shot=4,
             n_query=5, erase_p=0.5):
    """Same objective as train_heads, on an arbitrary subject set and embedding dict."""
    idx = {c: i for i, c in enumerate(cells)}
    nm = [ds.qidx(q) for q in COND_SEQS['nm']]
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    head = VarHead(E['clean'].shape[2], C=E['clean'].shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for it in range(iters):
        ids = rng.choice(subjects, n_way, replace=False)
        sup, qry, lab = [], [], []
        for c, s_ in enumerate(ids):
            g = [(s_, q, v) for q in nm for v in range(11) if (s_, q, v) in idx]
            sc = [g[i] for i in rng.choice(len(g), k_shot, replace=False)]
            used = {x[2] for x in sc}
            p = [(s_, q, v) for q in range(10) for v in range(11)
                 if (s_, q, v) in idx and v not in used and (s_, q, v) not in sc]
            qc = [p[i] for i in rng.choice(len(p), min(n_query, len(p)), replace=False)]
            sup += sc; qry += qc; lab += [c] * len(qc)

        def pick(cs):
            keys = ['clean' if rng.rand() >= erase_p else occ_keys[rng.randint(len(occ_keys))] for _ in cs]
            return torch.stack([E[k][idx[c]] for k, c in zip(keys, cs)])
        Ms, Mq = pick(sup), pick(qry)
        vs, vq = head(Ms), head(Mq)
        owner = torch.arange(n_way, device=device).repeat_interleave(k_shot)
        mc, vc = fuse(Ms, vs, owner, n_way)
        y = torch.tensor(lab, device=device)
        s = vq + vc[y]
        loss = (0.5 * (((Mq - mc[y]) ** 2).sum(1) / s + Mq.shape[1] * torch.log(s))).mean() / Mq.shape[1]
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def crossfit(device='cuda', seeds=(0, 1, 2)):
    """Bias check: fit the head on half of the UNSEEN test subjects, evaluate on the
    other half (and swap). Occluded fitting samples use the test occlusion embeddings
    of the FITTING fold only, so no evaluation cell is ever used for fitting."""
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    folds = [test[0::2], test[1::2]]
    cells, E = torch.load('results/p5_frozen/embeddings.pt', weights_only=False)
    out = {}
    for bb in BACKBONES:
        Eb = {k: v.to(device) for k, v in E[bb].items()}
        out[bb] = {}
        for f in range(2):
            fit, ev = folds[f], folds[1 - f]
            base = evaluate_backbone(bb, cells, E[bb], None, ds, ev, device, kinds=['frozen'])['frozen']
            out[bb][f'fold{f}'] = {'frozen': base, 'n1n2': {}}
            for seed in seeds:
                h = fit_head(cells, Eb, fit, seed, ds, device, ['occ1', 'occ2'])
                out[bb][f'fold{f}']['n1n2'][seed] = evaluate_backbone(bb, cells, E[bb], h, ds, ev, device,
                                                                     kinds=['n1n2'])['n1n2']
                print(bb, 'fold', f, 'seed', seed, 'done', flush=True)
            save_json(out, f'{OUT}/crossfit.json')


MODES = ['nm', 'bg', 'cl', 'nm_occ']


def select_rule(r, rules):
    """Rule with the best mean OSCR over probe modes (computed on the calibration fold)."""
    return max(rules, key=lambda x: np.mean([r[f'os_{m}/{x}/oscr'] for m in MODES]))


def test_embeddings(bbs):
    """Clean/occ1/occ2 test embeddings: the original cache, or the per-variant files of frozen_hybrid."""
    if all(os.path.exists(f'results/p5_frozen/hyb_{bb}_{k}.pt') for bb in bbs for k in ['clean', 'occ1', 'occ2']):
        from .frozen_hybrid import cells_and_ds
        cells = cells_and_ds()[2]
        return cells, {bb: {k: torch.load(f'results/p5_frozen/hyb_{bb}_{k}.pt').float() for k in ['clean', 'occ1', 'occ2']}
                       for bb in bbs}
    return torch.load('results/p5_frozen/embeddings.pt', weights_only=False)


def final_protocol(device='cuda', seeds=(0, 1, 2, 3, 4), bbs=BACKBONES):
    """FINAL plug-in protocol (frozen official checkpoints, no backbone training):
    the 50 LT test subjects (never seen by the backbones) are split into two
    folds. For each fold f: the head is fitted on the CALIBRATION fold, the
    open-set score rule is chosen on the CALIBRATION fold (frozen: raw|margin;
    head: raw|margin|lr; best mean OSCR over probe modes), and every metric is
    reported on the other (EVALUATION) fold. Then swap folds."""
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    folds = [test[0::2], test[1::2]]
    cells, E = test_embeddings(bbs)
    path = f'{OUT}/final_protocol.json'
    out = json.load(open(path)) if os.path.exists(path) else {}
    for bb in bbs:
        Eb = {k: v.to(device) for k, v in E[bb].items()}
        out[bb] = {}
        for f in range(2):
            cal, ev = folds[f], folds[1 - f]
            fr_cal = evaluate_backbone(bb, cells, E[bb], None, ds, cal, device, ['frozen'], closed=False)['frozen']
            fr_ev = evaluate_backbone(bb, cells, E[bb], None, ds, ev, device, ['frozen'])['frozen']
            fr_rule = select_rule(fr_cal, ['raw', 'margin'])
            out[bb][f'fold{f}'] = {'frozen': {'rule': fr_rule, 'eval': fr_ev}, 'head': {}}
            for seed in seeds:
                h = fit_head(cells, Eb, cal, seed, ds, device, ['occ1', 'occ2'])
                hc = evaluate_backbone(bb, cells, E[bb], h, ds, cal, device, ['n1n2'], closed=False)['n1n2']
                he = evaluate_backbone(bb, cells, E[bb], h, ds, ev, device, ['n1n2'])['n1n2']
                rule = select_rule(hc, ['raw', 'margin', 'lr'])
                out[bb][f'fold{f}']['head'][seed] = {'rule': rule, 'eval': he}
                print(bb, 'fold', f, 'seed', seed, 'rule', rule, '| frozen rule', fr_rule, flush=True)
                save_json(out, path)


if __name__ == '__main__':
    if sys.argv[1] == 'final' and len(sys.argv) > 2:          # final <device> <bb> [<bb> ...]
        final_protocol(device=sys.argv[2], bbs=sys.argv[3:])
    else:
        {'embed': embed_train, 'train': train_heads, 'eval': eval_all, 'crossfit': crossfit,
         'final': final_protocol}[sys.argv[1]]()
