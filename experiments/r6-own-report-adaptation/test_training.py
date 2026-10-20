import unittest
import numpy as np
import torch
from torch.nn import functional as F
from train import gates, state_losses


class TrainingContractTests(unittest.TestCase):
    def test_positive_negative_and_drift_gates(self):
        before = np.ones((64, 12)); after = before - .03
        p = np.full((64, 12), .5)
        self.assertTrue(gates(before, after, p, p)["passed"])
        self.assertFalse(gates(after, before, p, p)["passed"])
        self.assertFalse(gates(before, after, p, p + .4)["passed"])

    def test_unknown_states_are_not_negative_labels(self):
        prob = np.full((1, 12, 8), 1/8)
        target = [[-1]*12]; target[0][0] = 2
        result = state_losses(prob, target)
        self.assertEqual(np.isfinite(result).sum(), 1)
        self.assertAlmostEqual(result[0, 0], np.log(8))

    def test_invalid_source_padding_has_zero_finite_kl(self):
        a = torch.tensor([[[1., 2., float('-inf')]]], requires_grad=True)
        valid = torch.tensor([[True, True, False]])
        logp = a.log_softmax(-1).masked_fill(~valid[None], 0.)
        loss = F.kl_div(logp, a.detach().softmax(-1), reduction='none').sum()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())


if __name__ == "__main__":
    unittest.main()
