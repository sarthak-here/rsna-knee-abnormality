import unittest
import torch
from strict_reconstruct import last_stage_only


class Dummy(torch.nn.Module):
    def __init__(self, stages=4):
        super().__init__()
        self.enc = torch.nn.Module()
        self.enc.stages = torch.nn.Sequential(*[torch.nn.Linear(2, 2) for _ in range(stages)])
        self.state_head = torch.nn.Linear(2, 3)


class FreezeScopeTests(unittest.TestCase):
    def test_only_last_encoder_stage_trainable_and_weights_unchanged(self):
        model = Dummy()
        original = {k: v.clone() for k, v in model.state_dict().items()}
        names = last_stage_only(model)
        self.assertEqual(names, ["enc.stages.3.weight", "enc.stages.3.bias"])
        self.assertFalse(any(p.requires_grad for p in model.state_head.parameters()))
        self.assertTrue(all(torch.equal(original[k], v) for k, v in model.state_dict().items()))

    def test_incorrect_backbone_layout_fails(self):
        with self.assertRaises(ValueError):
            last_stage_only(Dummy(stages=3))


if __name__ == "__main__":
    unittest.main()
