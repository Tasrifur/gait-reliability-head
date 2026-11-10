"""Few-shot matching heads on the SHARED encoder (v0 + BatchNorm, 8 strips, 126k
params), for the Phase 5 controlled comparison. Only the head differs; the
encoder, data, episodic budget, optimiser and protocol are identical.

The strip embedding [P, D] is flattened to z in R^{P*D} for heads that treat
the embedding as a vector. Every head implements
    scores(S, owner, n, Q) -> logits [Nq, n]
for an arbitrary number of shots per class (owner[i] = class of support i), so
episodic training and full-gallery evaluation use the same code path.

  proto      Prototypical Networks (Snell 2017): -||q - c||^2 (flattened, i.e. the
             uniform part distance x P)
  matching   Matching Networks w/o FCE (Vinyals 2016): attention over individual
             supports, a = softmax(tau * cos(q, s_k)); logit_c = log sum_{k in c} a_k
  relation   Relation Networks (Sung 2018): MLP g([c ; q]) -> score
  metabase   Meta-Baseline (Chen 2021): cosine(q, c) * learnable tau; trained after
             a classifier pre-training stage (see train_heads.py)
  feat       FEAT (Ye 2020): one-layer self-attention transformer adapts the set of
             prototypes, then -||q - c'||^2 / temperature
  tadam      TADAM-style (Oreshkin 2018): learnable metric scale alpha and a
             task-conditioned FiLM (gamma, beta from the mean prototype) applied to
             query and prototype embeddings (FiLM on the embedding, not inside the
             CNN; a simplification)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .protonet import make_encoder


def class_means(S, owner, n):
    c = torch.zeros(n, S.shape[1], device=S.device).index_add_(0, owner, S)
    return c / torch.bincount(owner, minlength=n).clamp(min=1)[:, None].float()


class HeadNet(nn.Module):
    def __init__(self, head, num_parts=8):
        super().__init__()
        self.cnn, self.Dp = make_encoder('v0bn')
        self.P, self.head = num_parts, head
        d = self.P * self.Dp
        self.D = d
        if head in ('matching', 'metabase'):
            self.log_tau = nn.Parameter(torch.tensor(10.0).log())
        if head == 'relation':
            # LayerNorm on the pair: the raw (unnormalised) 1024-d embeddings saturate the MLP
            # and training stalls at chance; the original conv relation module used BatchNorm.
            self.rel = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 256), nn.ReLU(inplace=True),
                                     nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        if head == 'feat':
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
            self.v = nn.Linear(d, d, bias=False)
            self.ln = nn.LayerNorm(d)
            self.drop = nn.Dropout(0.1)
            self.log_temp = nn.Parameter(torch.tensor(64.0).log())
        if head == 'tadam':
            self.film = nn.Sequential(nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Linear(256, 2 * d))
            nn.init.zeros_(self.film[2].weight)
            nn.init.zeros_(self.film[2].bias)
            self.log_alpha = nn.Parameter(torch.tensor(1.0).log())
        if head == 'metabase':
            self.classifier = None          # created by train_heads for pre-training

    def embed(self, x):
        f = self.cnn(x)
        B, C, H, W = f.shape
        z = f.view(B, C, self.P, H // self.P, W).mean(dim=(3, 4)).transpose(1, 2)   # [B,P,C]
        return z.reshape(B, -1)

    def encoder(self, x):                     # [B,P,C] for compatibility
        return self.embed(x).view(x.shape[0], self.P, self.Dp)

    def scores(self, S, owner, n, Q):
        h = self.head
        if h == 'proto':
            c = class_means(S, owner, n)
            return -((Q[:, None] - c[None]) ** 2).sum(-1) / self.P
        if h == 'matching':
            a = torch.softmax(self.log_tau.exp() * F.normalize(Q, dim=-1) @ F.normalize(S, dim=-1).T, 1)
            p = torch.zeros(len(Q), n, device=Q.device).index_add_(1, owner, a)
            return torch.log(p.clamp(min=1e-12))
        if h == 'relation':
            c = class_means(S, owner, n)
            pair = torch.cat([c[None].expand(len(Q), -1, -1), Q[:, None].expand(-1, n, -1)], -1)
            return self.rel(pair).squeeze(-1)
        if h == 'metabase':
            c = class_means(S, owner, n)
            return self.log_tau.exp() * F.normalize(Q, dim=-1) @ F.normalize(c, dim=-1).T
        if h == 'feat':
            c = class_means(S, owner, n)
            att = torch.softmax(self.q(c) @ self.k(c).T / c.shape[1] ** 0.5, -1)
            c = self.ln(c + self.drop(att @ self.v(c)))
            return -((Q[:, None] - c[None]) ** 2).sum(-1) / self.log_temp.exp()
        if h == 'tadam':
            c = class_means(S, owner, n)
            g, b = self.film(c.mean(0)).chunk(2)
            c, Qf = c * (1 + g) + b, Q * (1 + g) + b
            return -self.log_alpha.exp() * ((Qf[:, None] - c[None]) ** 2).sum(-1) / self.P
        raise ValueError(h)

    def forward(self, support, query, n_way, k_shot):
        owner = torch.arange(n_way, device=support.device).repeat_interleave(k_shot)
        return self.scores(self.embed(support), owner, n_way, self.embed(query)), None

    def logits_unequal(self, support, owner, n_way, query, chunk=256):
        S = torch.cat([self.embed(support[i:i + chunk]) for i in range(0, len(support), chunk)])
        return torch.cat([self.scores(S, owner, n_way, self.embed(query[i:i + chunk]))
                          for i in range(0, len(query), chunk)])
