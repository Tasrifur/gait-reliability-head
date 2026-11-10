"""Phase 5b: frozen fully-supervised gait embeddings (OpenGait checkpoints) under
OUR protocols.

Models (CASIA-B LT checkpoints from github.com/ShiqiYu/OpenGait; trained on
subjects 001-074, so the 50 test subjects are unseen):
    GaitBase (GaitBase_DA-60000.pt, ResNet9 + 16-part HPP, 256-d per part)
    GaitSet  (GaitSet-40000.pt, 62-part HPP, 256-d per part)
Input: the full silhouette sequence of a cell (valid cells only, >=15 frames),
BaseSilCuttingTransform (64x64 -> 64x44, /255), set pooling over all frames.
Embeddings are cached for clean sequences and for the two deterministic
occlusion settings (the same band on every frame; same bands as the GEI runs).

Distance: OpenGait 'euc' = mean over parts of the per-part Euclidean distance.
Evaluations:
  published   : their CASIA-B protocol (gallery NM-01..04, nearest neighbour,
                identical view excluded) -> port sanity check vs. papers
  FG-CV       : our full-gallery cross-view protocol with PROTOTYPES (mean of
                the subject's gallery embeddings = few-shot enrollment)
  S1/S3/S4    : enrollment scenarios as in eval_enrollment
  open-set    : as in eval_openset (25/25 x 10 splits, raw and margin scores)
Deterministic models -> single numbers (no training seeds).

Usage: python -m gaitlab.external_eval
"""
import io
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .data.casiab import CasiaB, LT_GALLERY, LT_PROBES, SEQS, VIEWS
from .eval_enrollment import GAL, PRB, band_for
from .eval_openset import FARS, auroc, dir_at_far, oscr
from .utils import save_json

OG = os.path.join('external', 'opengait_src', 'opengait')
CK = os.path.join('external', 'opengait_ckpt', 'CASIA-B')


def build_opengait(name):
    """GaitPart / GaitGL from OpenGait's own model classes (network only, no BaseModel runtime)."""
    import types
    if 'modeling.models' not in sys.modules:             # skip models/__init__, which imports every model
        import modeling
        pkg = types.ModuleType('modeling.models')
        pkg.__path__ = [os.path.join(OG, 'modeling', 'models')]
        sys.modules['modeling.models'] = modeling.models = pkg
    if name == 'gaitpart':
        from modeling.models.gaitpart import GaitPart as cls
        cfg = {'backbone_cfg': {'type': 'Plain', 'in_channels': 1,
                                'layers_cfg': ['BC-32', 'BC-32', 'M', 'FC-64-2', 'FC-64-2', 'M',
                                               'FC-128-3', 'FC-128-3']},
               'SeparateFCs': {'in_channels': 128, 'out_channels': 128, 'parts_num': 16}, 'bin_num': [16]}
        ck = os.path.join(CK, 'GaitPart', 'GaitPart', 'checkpoints', 'GaitPart-120000.pt')
    else:
        from modeling.models.gaitgl import GaitGL as cls
        cfg = {'channels': [32, 64, 128], 'class_num': 74}
        ck = os.path.join(CK, 'GaitGL', 'GaitGL', 'checkpoints', 'GaitGL-80000.pt')

    class Net(cls):
        def __init__(self):
            nn.Module.__init__(self)
            self.cfgs = {'data_cfg': {'dataset_name': 'CASIA-B'}}
            self.build_network(cfg)

        def forward(self, sils):                        # [n, 1, s, h, w] -> [n, c, p]
            if name == 'gaitpart':
                out = self.TFA(self.HPP(self.Backbone(sils)), None)
                return self.Head(out)
            if sils.shape[2] < 3:
                sils = sils.repeat(1, 1, 3 if sils.shape[2] == 1 else 2, 1, 1)
            o = self.GLConvB2(self.GLConvA1(self.MaxPool0(self.GLConvA0(self.LTA(self.conv3d(sils))))))
            o = self.HPP(self.TP(o, seqL=None, options={'dim': 2})[0])
            return self.Bn(self.Head0(o))                 # OpenGait inference embedding (BN head)
    return Net(), ck


def build(name):
    sys.path.insert(0, OG)
    import modeling.modules as M
    from modeling.backbones.resnet import ResNet9
    if name in ('gaitpart', 'gaitgl'):
        m, ck = build_opengait(name)
    elif name == 'gaitbase':
        class GaitBase(nn.Module):
            def __init__(self):
                super().__init__()
                self.Backbone = M.SetBlockWrapper(ResNet9('BasicBlock', [64, 128, 256, 512], 1,
                                                          [1, 1, 1, 1], [1, 2, 2, 1], False))
                self.FCs = M.SeparateFCs(16, 512, 256)
                self.BNNecks = M.SeparateBNNecks(16, 256, 74)
                self.TP = M.PackSequenceWrapper(torch.max)
                self.HPP = M.HorizontalPoolingPyramid([16])

            def forward(self, sils):                    # [n, 1, s, h, w]
                o = self.TP(self.Backbone(sils), None, options={'dim': 2})[0]
                return self.FCs(self.HPP(o))           # [n, 256, 16]
        m, ck = GaitBase(), os.path.join(CK, 'Baseline', 'GaitBase_DA', 'checkpoints', 'GaitBase_DA-60000.pt')
    else:
        import copy

        class GaitSet(nn.Module):
            def __init__(self):
                super().__init__()
                c = [1, 32, 64, 128]
                blk = lambda a, b, k, p, pool: [M.BasicConv2d(a, b, k, 1, p), nn.LeakyReLU(inplace=True),
                                                M.BasicConv2d(b, b, 3, 1, 1), nn.LeakyReLU(inplace=True)] + \
                                               ([nn.MaxPool2d(2, 2)] if pool else [])
                b1 = nn.Sequential(*blk(c[0], c[1], 5, 2, True))
                b2 = nn.Sequential(*blk(c[1], c[2], 3, 1, True))
                b3 = nn.Sequential(*blk(c[2], c[3], 3, 1, False))
                self.gl_block2, self.gl_block3 = copy.deepcopy(b2), copy.deepcopy(b3)
                self.set_block1, self.set_block2, self.set_block3 = (M.SetBlockWrapper(b1),
                                                                     M.SetBlockWrapper(b2), M.SetBlockWrapper(b3))
                self.set_pooling = M.PackSequenceWrapper(torch.max)
                self.Head = M.SeparateFCs(62, 128, 256)
                self.HPP = M.HorizontalPoolingPyramid([16, 8, 4, 2, 1])

            def forward(self, sils):
                sp = lambda x: self.set_pooling(x, None, options={'dim': 2})[0]
                o = self.set_block1(sils)
                gl = self.gl_block2(sp(o))
                o = self.set_block2(o)
                gl = self.gl_block3(gl + sp(o))
                o = self.set_block3(o)
                o = sp(o)
                gl = gl + o
                return self.Head(torch.cat([self.HPP(o), self.HPP(gl)], -1))
        m, ck = GaitSet(), os.path.join(CK, 'GaitSet', 'GaitSet', 'checkpoints', 'GaitSet-40000.pt')
    sd = torch.load(ck, map_location='cpu', weights_only=True)['model']
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, missing
    assert all(k.startswith('BNNecks.fc_bin') or 'BNNecks' in k for k in unexpected) or not unexpected, unexpected
    return m.eval()


@torch.no_grad()
def embed_all(models, ds, test, zpath, device):
    zf = zipfile.ZipFile(zpath)
    groups = defaultdict(list)
    for n in zf.namelist():
        if n.endswith('.png'):
            s, q, v, _ = n.split('/')
            groups[(s, q, v)].append(n)
    cells = [(s, q, v) for s in test for q in range(10) for v in range(11) if ds.valid[s, q, v]]
    out = {name: {k: [] for k in ['clean', 'occ1', 'occ2']} for name in models}
    for i, (s, q, v) in enumerate(cells):
        files = sorted(groups[(ds.subjects[s], SEQS[q], VIEWS[v])])
        fr = np.stack([np.asarray(Image.open(io.BytesIO(zf.read(f))).convert('L'), np.float32)
                       for f in files])
        fr = fr[(fr > 127).sum((1, 2)) >= 50][:720]          # same frame filter as the GEI cache
        variants = {'clean': fr}
        for key, salt, p in [('occ1', 1, 0.5), ('occ2', 2, 1.0)]:
            b = band_for((s, q, v), salt, p)
            f2 = fr.copy()
            if b is not None:
                f2[:, b[0]:b[0] + b[1], :] = 0
            variants[key] = f2
        for key, x in variants.items():
            t = torch.from_numpy(x[:, :, 10:-10] / 255.0)[None, None].to(device)   # [1,1,s,64,44]
            for name, m in models.items():
                out[name][key].append(m(t)[0].cpu())
        if i % 500 == 0:
            print(f'{i}/{len(cells)} sequences embedded', flush=True)
    return cells, {n: {k: torch.stack(v) for k, v in d.items()} for n, d in out.items()}


def dist(a, b):
    """OpenGait euc: mean over parts of per-part Euclidean distance. a [n,c,p], b [m,c,p]."""
    return torch.sqrt(((a[:, None] - b[None]) ** 2).sum(2).clamp(min=0)).mean(-1)


def evaluate(cells, E, ds, test, device):
    idx = {c: i for i, c in enumerate(cells)}
    get = lambda key, cs: E[key][[idx[c] for c in cs]].to(device)
    gq = [ds.qidx(x) for x in LT_GALLERY]
    res = {}

    # published protocol: nearest neighbour over individual gallery sequences
    for cond in ['nm', 'bg', 'cl']:
        pq = [ds.qidx(x) for x in LT_PROBES[cond]]
        accs = []
        for pv in range(11):
            per = []
            for gv in range(11):
                if gv == pv:
                    continue
                g = [(s, x, gv) for s in test for x in gq if (s, x, gv) in idx]
                p = [(s, x, pv) for s in test for x in pq if (s, x, pv) in idx]
                d = dist(get('clean', p), get('clean', g))
                pred = np.array([g[j][0] for j in d.argmin(1).cpu().numpy()])
                per.append((pred == np.array([c[0] for c in p])).mean() * 100)
            accs.append(np.mean(per))
        res[f'published_nn>{cond}'] = float(np.mean(accs))

    def fg(gseqs, pseqs, gkey='clean', pkey='clean'):
        gq_, pq_ = [ds.qidx(x) for x in gseqs], [ds.qidx(x) for x in pseqs]
        accs = []
        for pv in range(11):
            per = []
            for gv in range(11):
                if gv == pv:
                    continue
                ids = [s for s in test if any((s, x, gv) in idx for x in gq_)]
                proto = torch.stack([get(gkey, [(s, x, gv) for x in gq_ if (s, x, gv) in idx]).mean(0)
                                     for s in ids])
                p = [(s, x, pv) for s in ids for x in pq_ if (s, x, pv) in idx]
                if not p:
                    continue
                pred = dist(get(pkey, p), proto).argmin(1).cpu().numpy()
                per.append((np.array([ids[j] for j in pred]) == np.array([c[0] for c in p])).mean() * 100)
            accs.append(np.mean(per))
        return float(np.mean(accs))

    for cond in ['nm', 'bg', 'cl']:
        res[f'fgcv>{cond}'] = fg(LT_GALLERY, LT_PROBES[cond])
        res[f'S3_corrupt>{cond}'] = fg(LT_GALLERY, LT_PROBES[cond], gkey='occ1')
        res[f'S4_occprobe>{cond}'] = fg(LT_GALLERY, LT_PROBES[cond], pkey='occ2')
    for g in ['nm', 'bg', 'cl']:
        for p in ['nm', 'bg', 'cl']:
            res[f'S1_{g}>{p}'] = fg(GAL[g], PRB[p])

    # open-set, same splits as eval_openset
    for mode in ['nm', 'bg', 'cl', 'nm_occ']:
        cond = 'nm' if mode == 'nm_occ' else mode
        pkey = 'occ2' if mode == 'nm_occ' else 'clean'
        pq = [ds.qidx(x) for x in LT_PROBES[cond]]
        rng = np.random.RandomState(2024)
        reps = []
        for _ in range(10):
            enr = set(sorted(rng.choice(test, len(test) // 2, replace=False).tolist()))
            sk, su, ck = {'raw': [], 'margin': []}, {'raw': [], 'margin': []}, []
            for gv in range(11):
                ids = [s for s in sorted(enr) if any((s, x, gv) in idx for x in gq)]
                proto = torch.stack([get('clean', [(s, x, gv) for x in gq if (s, x, gv) in idx]).mean(0)
                                     for s in ids])
                for pv in range(11):
                    if pv == gv:
                        continue
                    p = [(s, x, pv) for s in test for x in pq if (s, x, pv) in idx]
                    lg = -dist(get(pkey, p), proto)
                    t2 = lg.topk(2, 1)
                    known = np.array([c[0] in enr for c in p])
                    sc = {'raw': t2.values[:, 0].cpu().numpy(),
                          'margin': (t2.values[:, 0] - t2.values[:, 1]).cpu().numpy()}
                    am = t2.indices[:, 0].cpu().numpy()
                    ck += [ids[a] == c[0] for c, a, k in zip(p, am, known) if k]
                    for r in sc:
                        sk[r].append(sc[r][known]); su[r].append(sc[r][~known])
            ck = np.array(ck)
            o = {}
            for r in sk:
                k_, u_ = np.concatenate(sk[r]), np.concatenate(su[r])
                o[f'{r}/auroc'], o[f'{r}/oscr'] = auroc(k_, u_), oscr(k_, ck, u_)
                for f in FARS:
                    o[f'{r}/dir@far{int(f * 100)}'] = dir_at_far(k_, ck, u_, f)
            reps.append(o)
        for k in reps[0]:
            res[f'os_{mode}/{k}'] = float(np.mean([r[k] for r in reps]))
    return res


def main():
    device = 'cuda'
    ds = CasiaB('data/cache/casiab_gei.npz')
    test = ds.lt_split(12)['test']
    models = {n: build(n).to(device) for n in ['gaitbase', 'gaitset']}
    for n, m in models.items():
        print(n, 'params', sum(p.numel() for p in m.parameters()), flush=True)
    cache = 'results/p5_frozen/embeddings.pt'
    if os.path.exists(cache):
        cells, E = torch.load(cache, weights_only=False)
    else:
        cells, E = embed_all(models, ds, test, 'casia_b_silhouettes.zip', device)
        os.makedirs('results/p5_frozen', exist_ok=True)
        torch.save((cells, E), cache)
    out = {}
    for n in models:
        out[n] = evaluate(cells, E[n], ds, test, device)
        out[n]['n_params'] = sum(p.numel() for p in models[n].parameters())
        r = out[n]
        print(f"{n}: published-protocol NN rank-1 NM {r['published_nn>nm']:.2f} BG {r['published_nn>bg']:.2f} "
              f"CL {r['published_nn>cl']:.2f} | FG-CV proto NM {r['fgcv>nm']:.2f} BG {r['fgcv>bg']:.2f} "
              f"CL {r['fgcv>cl']:.2f}", flush=True)
    save_json(out, 'results/p5_frozen/results.json')


if __name__ == '__main__':
    main()
