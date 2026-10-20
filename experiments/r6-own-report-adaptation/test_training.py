import unittest
import numpy as np
import torch
from torch.nn import functional as F
from train import gates, state_losses, training_objective, AMP_INITIAL_SCALE


class TrainingContractTests(unittest.TestCase):
    def test_shared_objective_and_numerical_scale(self):
        self.assertEqual(AMP_INITIAL_SCALE, 1024.)
        z = torch.zeros(1, 12, requires_grad=True)
        sl = torch.zeros(1, 12, 3, requires_grad=True)
        bl = torch.zeros(1, 2, requires_grad=True)
        valid = torch.ones(12, 3, dtype=torch.bool)
        targets = torch.full((12,), -1); targets[:4] = 0
        counts = np.zeros(12); counts[:4] = 48
        refs = (z.detach().clone(), sl.detach().clone(), bl.detach().clone())
        loss = training_objective((z, sl, bl), refs, targets, counts, valid)
        self.assertAlmostEqual(float(loss.detach()), 4*np.log(3), places=5)
        loss.backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in (z, sl, bl)))

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
