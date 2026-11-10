"""Probabilistic part prototypes (Phase 4: N1 + N2), built on the B1 backbone.

Each strip i of a GEI gets a mean embedding mu_i in R^D and a scalar variance
s_i > 0 (isotropic within the strip). The variance head sees the strip's
spatial mean and spatial std (blur/smear inside the strip signals unreliability).

N1  distance between query q and prototype c (Gaussian negative log-likelihood
    of the difference mu_q - mu_c ~ N(0, (s_q + s_c) I_D), summed over strips):
        d(q, c) = sum_i [ ||mu_q,i - mu_c,i||^2 / (s_q,i + s_c,i)
                          + lam * D * log(s_q,i + s_c,i) ] / 2
    The effective weight of strip i is 1/(s_q,i + s_c,i): it depends on BOTH
    sides of the comparison, so it cannot be absorbed into the encoder.
    lam = 1 is the exact likelihood; lam < 1 down-weights the log term.

N2  prototype from K supports with precision weighting (closed-form Gaussian
    product; constant-time, incremental enrollment):
        s_c,i  = 1 / sum_k (1/s_k,i)
        mu_c,i = s_c,i * sum_k mu_k,i / s_k,i
    With fusion='mean', the prototype is the plain mean and s_c = mean_k s_k
    (N1 only).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .protonet import make_encoder


class ProbProtoNet(nn.Module):
    def __init__(self, encoder='v0bn', width=1.0, num_parts=8, fusion='precision',
                 logdet_weight=1.0, var_init=4.0, var_floor=1e-3):
        super().__init__()
        self.cnn, self.D = make_encoder(encoder, width)
        self.P, self.fusion, self.lam, self.floor = num_parts, fusion, logdet_weight, var_floor
        self.var_head = nn.Sequential(nn.Linear(2 * self.D, 64), nn.ReLU(inplace=True),
                                      nn.Linear(64, 1))
        nn.init.zeros_(self.var_head[2].weight)
        inv_sp = torch.log(torch.expm1(torch.tensor(var_init - var_floor)))
        self.part_bias = nn.Parameter(torch.full((num_parts,), float(inv_sp)))
        nn.init.zeros_(self.var_head[2].bias)

    # ---- embedding ---------------------------------------------------------
    def embed(self, x):
        """[B,1,64,64] -> mu [B,P,D], var [B,P]."""
        f = self.cnn(x)
        B, C, H, W = f.shape
        g = f.view(B, C, self.P, H // self.P, W).flatten(3)          # [B,C,P,hw]
        mu = g.mean(-1).transpose(1, 2)                                # [B,P,C]
        sd = g.std(-1).transpose(1, 2)
        h = self.var_head(torch.cat([mu, sd], -1)).squeeze(-1)        # [B,P]
        var = F.softplus(h + self.part_bias) + self.floor
        return mu, var

    def encoder(self, x):                       # mean embedding only (diagnostics)
        return self.embed(x)[0]

    # ---- prototypes ----------------------------------------------------------
    def fuse(self, mu, var, owner, n_way):
        """mu [M,P,D], var [M,P], owner [M] -> proto mu [N,P,D], var [N,P]."""
        if self.fusion == 'precision':
            prec = 1.0 / var
            psum = torch.zeros(n_way, self.P, device=mu.device).index_add_(0, owner, prec)
            msum = torch.zeros(n_way, self.P, self.D, device=mu.device).index_add_(
                0, owner, mu * prec[..., None])
            return msum / psum[..., None], 1.0 / psum
        cnt = torch.bincount(owner, minlength=n_way).clamp(min=1).float()
        m = torch.zeros(n_way, self.P, self.D, device=mu.device).index_add_(0, owner, mu)
        v = torch.zeros(n_way, self.P, device=mu.device).index_add_(0, owner, var)
        return m / cnt[:, None, None], v / cnt[:, None]

    # ---- distances -------------------------------------------------------------
    def part_nll(self, mq, vq, mc, vc):
        """Per-strip NLL terms [Nq, N, P]."""
        s = vq[:, None] + vc[None]                                    # [Nq,N,P]
        sq = ((mq[:, None] - mc[None]) ** 2).sum(-1)                  # [Nq,N,P]
        return 0.5 * (sq / s + self.lam * self.D * torch.log(s))

    def logits_from_embed(self, mq, vq, mc, vc, return_part=False):
        nll = self.part_nll(mq, vq, mc, vc)
        logits = -nll.sum(-1)
        return (logits, -nll) if return_part else logits

    def forward(self, support, query, n_way, k_shot, return_part_logits=False):
        ms, vs = self.embed(support)
        mq, vq = self.embed(query)
        owner = torch.arange(n_way, device=support.device).repeat_interleave(k_shot)
        mc, vc = self.fuse(ms, vs, owner, n_way)
        self._last = (mq, vq, mc, vc)                                 # for the MLS term in training
        w = 1.0 / (vq[:, None] + vc[None])                            # effective strip weights
        w = (w / w.sum(-1, keepdim=True)).mean(1)                     # [Nq,P] (for logging only)
        if return_part_logits:
            logits, part = self.logits_from_embed(mq, vq, mc, vc, True)
            return logits, w, part
        return self.logits_from_embed(mq, vq, mc, vc), w

    def pos_nll_detached_means(self, y):
        """Positive-pair per-strip NLL [Nq, P] from the last forward pass, with the
        MEANS detached: trains only the variance head (PFE-style), so the term
        cannot be lowered by shrinking the embedding scale."""
        mq, vq, mc, vc = self._last
        mcy, vcy = mc.detach()[y], vc[y]
        s = vq + vcy
        sq = ((mq.detach() - mcy) ** 2).sum(-1)
        return 0.5 * (sq / s + self.lam * self.D * torch.log(s))

    def logits_unequal(self, support, owner, n_way, query, chunk=256):
        outs = [self.embed(support[i:i + chunk]) for i in range(0, len(support), chunk)]
        ms, vs = torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
        mc, vc = self.fuse(ms, vs, owner, n_way)
        res = []
        for i in range(0, len(query), chunk):
            mq, vq = self.embed(query[i:i + chunk])
            res.append(self.logits_from_embed(mq, vq, mc, vc))
        return torch.cat(res)
