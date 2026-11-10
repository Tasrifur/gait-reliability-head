"""Few-shot head checks. Run: PYTHONPATH=src python -m pytest tests -q"""
import pytest
import torch

from gaitlab.models.heads import HeadNet


@pytest.mark.parametrize('head', ['proto', 'matching', 'relation', 'metabase', 'feat', 'tadam'])
def test_shapes_and_unequal_path(head):
    torch.manual_seed(0)
    m = HeadNet(head).eval()
    S, Q = torch.rand(12, 1, 64, 64), torch.rand(5, 1, 64, 64)
    a, _ = m(S, Q, 3, 4)
    assert a.shape == (5, 3) and torch.isfinite(a).all()
    b = m.logits_unequal(S, torch.arange(3).repeat_interleave(4), 3, Q)
    assert torch.allclose(a, b, atol=1e-4)


def test_proto_head_equals_uniform_protonet():
    from gaitlab.models.protonet import ProtoNet
    torch.manual_seed(0)
    h = HeadNet('proto').eval()
    p = ProtoNet('v0bn').eval()
    p.cnn.load_state_dict(h.cnn.state_dict())
    S, Q = torch.rand(8, 1, 64, 64), torch.rand(3, 1, 64, 64)
    assert torch.allclose(h(S, Q, 2, 4)[0], p(S, Q, 2, 4)[0], atol=1e-4)
