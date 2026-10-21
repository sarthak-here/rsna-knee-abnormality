"""Zero-initialized rank8 updates inside the original final ConvNeXt blocks."""
import math
import torch
from torch import nn

RANK = 8


class LowRankLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        if not isinstance(base, nn.Linear) or (base.in_features, base.out_features) != (3072, 768):
            raise ValueError("expected original ConvNeXt-Small final-block fc2")
        self.base = base.requires_grad_(False)
        self.adapter_down = nn.Linear(3072, RANK, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.adapter_up = nn.Linear(RANK, 768, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, x):
        return self.base(x) + self.adapter_up(self.adapter_down(x))


def install(model):
    model.eval().requires_grad_(False)
    stage = model.enc.stages[3]
    if len(stage.blocks) != 3:
        raise ValueError("expected three final-stage blocks")
    for block in stage.blocks:
        block.mlp.fc2 = LowRankLinear(block.mlp.fc2)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if len(names) != 6 or any(not n.startswith("enc.stages.3.blocks.") or ".adapter_" not in n for n in names):
        raise ValueError("adapter trainable scope mismatch")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 92160:
        raise ValueError("adapter parameter count mismatch")
    return names


def assert_frozen(reference, candidate):
    """Preserve every original tensor, remapping only fc2.base names."""
    state = candidate.state_dict()
    for name, tensor in reference.state_dict().items():
        key = name.replace(".mlp.fc2.", ".mlp.fc2.base.") if name.startswith("enc.stages.3.blocks.") else name
        if key not in state or tensor.dtype != state[key].dtype or not torch.equal(tensor, state[key]):
            raise RuntimeError("original tensor changed: " + name)
    adapters = [n for n in state if ".adapter_" in n]
    if len(adapters) != 6 or len(state) != len(reference.state_dict()) + 6:
        raise RuntimeError("unexpected added/missing tensor")
    return adapters
