"""DPP-Net v0: faithful port of the original notebook model (CASIA_B.ipynb, cell 3).

Kept unchanged so that re-baselining under the fixed protocol measures the
protocol effect only. `attention` selects ablation variants:
    'residual' : MLP([q ; q - mean_proto])   (original)
    'concat'   : MLP([q ; mean_proto])
    'query'    : MLP(q)
    'global'   : learned weight vector, same for every query
    'uniform'  : w_i = 1/P (no parameters)
"""
import torch
import torch.nn as nn


class PartAwareGaitEncoder(nn.Module):
    def __init__(self, in_ch=1, hidden_dim=128, num_parts=8):
        super().__init__()
        self.num_parts = num_parts
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        f = self.cnn(x)                                   # [B, C, H, W]
        B, C, H, W = f.shape
        ph = H // self.num_parts
        parts = [f[:, :, i * ph:(i + 1) * ph, :].mean(dim=(2, 3))
                 for i in range(self.num_parts)]
        return torch.stack(parts, dim=1)                  # [B, P, C]


class DPPNet(nn.Module):
    def __init__(self, hidden_dim=128, num_parts=8, attention='residual'):
        super().__init__()
        self.encoder = PartAwareGaitEncoder(1, hidden_dim, num_parts)
        self.P, self.D, self.attention = num_parts, hidden_dim, attention
        in_dim = {'residual': 2, 'concat': 2, 'query': 1}.get(attention)
        if in_dim is not None:
            self.attention_net = nn.Sequential(
                nn.Linear(in_dim * hidden_dim * num_parts, 256),
                nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, num_parts))
        elif attention == 'global':
            self.global_logits = nn.Parameter(torch.zeros(num_parts))
        elif attention != 'uniform':
            raise ValueError(attention)

    def part_weights(self, Q, proto):
        n = Q.size(0)
        if self.attention == 'uniform':
            return Q.new_full((n, self.P), 1.0 / self.P)
        if self.attention == 'global':
            return torch.softmax(self.global_logits, 0).expand(n, -1)
        mean_proto = proto.mean(0, keepdim=True).expand_as(Q)
        if self.attention == 'residual':
            x = torch.cat([Q, Q - mean_proto], -1)
        elif self.attention == 'concat':
            x = torch.cat([Q, mean_proto], -1)
        else:
            x = Q
        return torch.softmax(self.attention_net(x.reshape(n, -1)), 1)

    def logits_from_proto(self, Q, proto, use_attention=True):
        w = self.part_weights(Q, proto)
        if not use_attention:                                      # legacy test-time ablation
            w = torch.full_like(w, 1.0 / self.P)
        dist = ((Q[:, None] - proto[None]) ** 2).sum(-1)           # [Nq, N, P]
        return -(dist * w[:, None]).sum(-1), w

    def forward(self, support, query, n_way, k_shot, use_attention=True):
        S, Q = self.encoder(support), self.encoder(query)
        proto = S.view(n_way, k_shot, self.P, self.D).mean(1)      # [N, P, D]
        return self.logits_from_proto(Q, proto, use_attention)

    def logits_unequal(self, support, owner, n_way, query, chunk=256):
        """Prototypes from a variable number of shots per class.
        owner[i] = class index of support[i]."""
        S = torch.cat([self.encoder(support[i:i + chunk])
                       for i in range(0, len(support), chunk)])
        proto = torch.zeros(n_way, self.P, self.D, device=S.device)
        proto.index_add_(0, owner, S)
        proto /= torch.bincount(owner, minlength=n_way).clamp(min=1)[:, None, None].float()
        out = [self.logits_from_proto(self.encoder(query[i:i + chunk]), proto)[0]
               for i in range(0, len(query), chunk)]
        return torch.cat(out)
