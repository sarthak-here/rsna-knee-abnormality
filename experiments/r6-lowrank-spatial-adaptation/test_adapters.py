import copy
import unittest
import torch
from torch import nn
from adapters import LowRankLinear, install, assert_frozen


def fixture():
    model = nn.Module()
    model.enc = nn.Module()
    stage = nn.Module()
    stage.blocks = nn.ModuleList()
    for _ in range(3):
        block = nn.Module(); block.mlp = nn.Module()
        block.mlp.fc2 = nn.Linear(3072, 768)
        stage.blocks.append(block)
    model.enc.stages = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity(), stage])
    model.cls_w = nn.Parameter(torch.ones(12, 768))
    return model.eval().requires_grad_(False)


class AdapterTests(unittest.TestCase):
    def test_zero_init_and_gradient_scope(self):
        torch.manual_seed(22)
        reference = fixture(); candidate = copy.deepcopy(reference)
        names = install(candidate)
        self.assertEqual(len(names), 6)
        x = torch.randn(4, 3072)
        old = reference.enc.stages[3].blocks[0].mlp.fc2(x)
        new = candidate.enc.stages[3].blocks[0].mlp.fc2(x)
        self.assertTrue(torch.equal(old, new))
        new.square().mean().backward()
        block = candidate.enc.stages[3].blocks[0].mlp.fc2
        self.assertTrue(torch.isfinite(block.adapter_up.weight.grad).all())
        self.assertGreater(block.adapter_up.weight.grad.abs().max(), 0)
        self.assertIsNone(block.base.weight.grad)
        self.assertEqual(len(assert_frozen(reference, candidate)), 6)

    def test_original_mutation_rejected(self):
        reference = fixture(); candidate = copy.deepcopy(reference); install(candidate)
        with torch.no_grad():
            candidate.cls_w[0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "original tensor changed"):
            assert_frozen(reference, candidate)

    def test_wrong_layer_rejected(self):
        with self.assertRaises(ValueError):
            LowRankLinear(nn.Linear(10, 4))

    def test_no_dropout_or_changed_base_training(self):
        reference = fixture(); install(reference)
        self.assertFalse(reference.training)
        self.assertFalse(reference.enc.stages[3].blocks[0].mlp.fc2.base.weight.requires_grad)
        self.assertEqual(sum(p.numel() for p in reference.parameters() if p.requires_grad), 92160)


if __name__ == "__main__":
    unittest.main()
