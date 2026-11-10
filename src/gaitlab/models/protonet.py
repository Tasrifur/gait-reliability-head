"""Configurable part-based prototypical network (Phase 3 backbone study).

Uniform part weighting throughout (Phase 2 showed learned weights add nothing
cross-view). Options:
  encoder : 'v0'   original 3-conv encoder (126k params)
            'v0bn' same + BatchNorm
            'res'  4-stage residual encoder with BatchNorm (~1.2M params)
  scales  : horizontal pooling pyramid, e.g. [8] (v0) or [1, 2, 4, 8]
  pool    : 'mean' (v0) or 'meanmax' (mean + max, as in GaitSet/GaitPart HPP)
  l2norm  : L2-normalise each strip before the distance
  scale   : temperature on logits; 'learn' = learnable (init `scale_init`), or a float
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = (nn.Sequential() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout)))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(self.b2(self.c2(y)) + self.short(x), inplace=True)


def make_encoder(name, width=1.0):
    if name == 'v0':
        net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        return net, 128
    if name == 'v0bn':
        net = nn.Sequential(conv_bn(1, 32, 5), nn.MaxPool2d(2),
                            conv_bn(32, 64, 5), nn.MaxPool2d(2),
                            conv_bn(64, 128, 3), nn.MaxPool2d(2))
        return net, 128
    if name == 'res':
        c = [int(round(x * width)) for x in (32, 64, 128, 256)]
        net = nn.Sequential(conv_bn(1, c[0], 3),                 # 64x64
                            BasicBlock(c[0], c[0], 1),
                            BasicBlock(c[0], c[1], 2),           # 32x32
                            BasicBlock(c[1], c[2], 2),           # 16x16
                            BasicBlock(c[2], c[3], 2))           # 8x8
        return net, c[3]
    raise ValueError(name)


class ProtoNet(nn.Module):
    def __init__(self, encoder='v0', width=1.0, scales=(8,), pool='mean',
                 l2norm=False, scale=1.0, scale_init=10.0):
        super().__init__()
        self.cnn, self.D = make_encoder(encoder, width)
        self.scales, self.pool, self.l2norm = list(scales), pool, l2norm
        self.P = sum(self.scales)
        if scale == 'learn':
            self.log_scale = nn.Parameter(torch.tensor(float(scale_init)).log())
        else:
            self.register_buffer('log_scale', torch.tensor(float(scale)).log())

    def encoder(self, x):
        """[B,1,64,64] -> [B, P, D] strip embeddings (pyramid concatenated)."""
        f = self.cnn(x)
        B, C, H, W = f.shape
        out = []
        for s in self.scales:
            g = f.view(B, C, s, H // s, W)
            v = g.mean(dim=(3, 4))
            if self.pool == 'meanmax':
                v = v + g.amax(dim=(3, 4))
            out.append(v.transpose(1, 2))              # [B, s, C]
        z = torch.cat(out, 1)
        return F.normalize(z, dim=-1) if self.l2norm else z

    def part_distances(self, Q, proto):
        return ((Q[:, None] - proto[None]) ** 2).sum(-1)          # [Nq, N, P]

    def logits_from_proto(self, Q, proto, use_attention=True):
        d = self.part_distances(Q, proto)
        w = Q.new_full((Q.size(0), self.P), 1.0 / self.P)
        return -self.log_scale.exp() * d.mean(-1), w

    def part_weights(self, Q, proto):          # for diagnostics compatibility
        return Q.new_full((Q.size(0), self.P), 1.0 / self.P)

    def forward(self, support, query, n_way, k_shot, return_part_logits=False):
        S, Q = self.encoder(support), self.encoder(query)
        proto = S.view(n_way, k_shot, self.P, self.D).mean(1)
        d = self.part_distances(Q, proto)
        s = self.log_scale.exp()
        logits = -s * d.mean(-1)
        w = Q.new_full((Q.size(0), self.P), 1.0 / self.P)
        if return_part_logits:
            return logits, w, -s * d                                 # [Nq, N, P]
        return logits, w

    def logits_unequal(self, support, owner, n_way, query, chunk=256):
        S = torch.cat([self.encoder(support[i:i + chunk]) for i in range(0, len(support), chunk)])
        proto = torch.zeros(n_way, self.P, self.D, device=S.device)
        proto.index_add_(0, owner, S)
        proto /= torch.bincount(owner, minlength=n_way).clamp(min=1)[:, None, None].float()
        return torch.cat([self.logits_from_proto(self.encoder(query[i:i + chunk]), proto)[0]
                          for i in range(0, len(query), chunk)])
