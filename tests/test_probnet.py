"""Math checks for ProbProtoNet (N1/N2). Run: PYTHONPATH=src python -m pytest tests -q"""
import torch

from gaitlab.models.probnet import ProbProtoNet


def test_precision_fusion_equal_variance_is_mean():
    m = ProbProtoNet(fusion='precision')
    mu = torch.randn(6, 8, m.D)
    var = torch.full((6, 8), 2.0)
    owner = torch.tensor([0, 0, 0, 1, 1, 1])
    mc, vc = m.fuse(mu, var, owner, 2)
    assert torch.allclose(mc[0], mu[:3].mean(0), atol=1e-5)
    assert torch.allclose(vc, torch.full((2, 8), 2.0 / 3), atol=1e-6)


def test_precision_fusion_closed_form():
    m = ProbProtoNet(fusion='precision')
    mu = torch.randn(2, 8, m.D)
    var = torch.tensor([[1.0] * 8, [3.0] * 8])
    mc, vc = m.fuse(mu, var, torch.tensor([0, 0]), 1)
    expect = (mu[0] / 1.0 + mu[1] / 3.0) / (1 / 1.0 + 1 / 3.0)
    assert torch.allclose(mc[0], expect, atol=1e-5)
    assert torch.allclose(vc[0], torch.full((8,), 0.75), atol=1e-6)


def test_constant_variance_ranks_like_uniform_distance():
    m = ProbProtoNet()
    mq, mc = torch.randn(10, 8, m.D), torch.randn(5, 8, m.D)
    vq, vc = torch.full((10, 8), 1.5), torch.full((5, 8), 0.5)
    logits = m.logits_from_embed(mq, vq, mc, vc)
    uniform = -((mq[:, None] - mc[None]) ** 2).sum(-1).mean(-1)
    assert torch.equal(logits.argsort(1), uniform.argsort(1))


def test_unequal_path_matches_forward():
    torch.manual_seed(0)
    m = ProbProtoNet().eval()
    S, Q = torch.rand(12, 1, 64, 64), torch.rand(5, 1, 64, 64)
    a, _ = m(S, Q, 3, 4)
    owner = torch.arange(3).repeat_interleave(4)
    b = m.logits_unequal(S, owner, 3, Q)
    assert torch.allclose(a, b, atol=1e-4)


def test_variance_positive_and_initialised():
    m = ProbProtoNet(var_init=4.0).eval()
    _, v = m.embed(torch.rand(3, 1, 64, 64))
    assert (v > 0).all() and torch.allclose(v, torch.full_like(v, 4.0), atol=1e-3)
