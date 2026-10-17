import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("pairwise_objective.py")
SPEC = importlib.util.spec_from_file_location("pairwise_objective", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_better_ordering_has_lower_loss():
    targets = torch.tensor([[0.0], [0.2], [0.8], [1.0]])
    ordered = torch.tensor([[-3.0], [-1.0], [1.0], [3.0]])
    reversed_order = ordered.flip(0)
    assert MODULE.weighted_soft_pairwise_auc(ordered, targets) < MODULE.weighted_soft_pairwise_auc(
        reversed_order, targets
    )


def test_multitarget_objective_is_finite_and_differentiable():
    generator = torch.Generator().manual_seed(20261123)
    logits = torch.randn(16, 12, generator=generator, requires_grad=True)
    targets = torch.rand(16, 12, generator=generator)
    loss = MODULE.pairwise_bce_objective(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_shape_and_threshold_fail_closed():
    try:
        MODULE.weighted_soft_pairwise_auc(torch.zeros(2, 2), torch.zeros(2, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch did not fail closed")

    try:
        MODULE.weighted_soft_pairwise_auc(torch.zeros(2, 1), torch.zeros(2, 1), threshold=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid threshold did not fail closed")
