"""New adapter graph only: synthetic GPU forward/backward, no optimizer."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from adapters import install, assert_frozen
from train import prefix, tail, training_objective, AMP_INITIAL_SCALE


def main():
    here = Path(__file__).resolve().parent; repo = here.parents[1]
    source = repo / ".audit/rsna-knee-blend-code/r6src"
    sys.path[:0] = [str(source), str(repo / ".audit/frontier-2026-10-17/goodpjw958/output/_own_env")]
    import compact_train as trainer
    torch.set_num_threads(2); torch.manual_seed(20261022)
    assets = repo / ".audit/r6-strict-load"
    path = assets / "E_lrprior_s0_f0.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != "59cc88a197bc80c7fa5faa89e69ae7bec04f69cab5259a23a0d0ceb9431e53f6":
        raise ValueError("checkpoint drift")
    spec = json.loads((assets / "state_spec.json").read_text())
    model = trainer.Net("convnext_small", None, crop=352, state_spec=spec, in_chans=3, spatial=True, extent=True, syn_decomp=True)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).cuda()
    candidate = copy.deepcopy(model)
    names = install(candidate)
    images = torch.randint(0, 256, (32, 3, 352, 352), dtype=torch.uint8, device="cuda")
    meta = torch.zeros((32, 9), device="cuda")
    meta[:, 1] = 1; meta[:, 3] = 1.5; meta[:, 4:6] = 1; meta[:, 8] = 32
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        early = torch.cat([prefix(model, images[i:i+8]) for i in range(0, 32, 8)]).half()
        teacher = tuple(x.float() for x in tail(model, early, meta))
        initial = tuple(x.float() for x in tail(candidate, early, meta))
    if not all(torch.equal(x, y) for x, y in zip(teacher, initial)):
        raise RuntimeError("zero-init adapter reader mismatch")
    target = torch.full((12,), -1, dtype=torch.long, device="cuda"); target[:4] = 0
    counts = np.zeros(12); counts[:4] = 48
    backward = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = tail(candidate, early.detach(), meta)
    loss = training_objective(outputs, teacher, target, counts, candidate.state_valid)
    (loss * AMP_INITIAL_SCALE).backward(); torch.cuda.synchronize()
    gradients = [(n, p.grad) for n, p in candidate.named_parameters() if p.requires_grad]
    if not torch.isfinite(loss) or len(gradients) != 6 or any(g is None or not torch.isfinite(g).all() for _, g in gradients):
        raise RuntimeError("nonfinite/missing rank8 scaled gradient")
    assert_frozen(model, candidate)
    result = {"status": "PASS", "synthetic_only": True, "optimizer_steps": 0,
              "checkpoint_sha256": digest, "initial_reader_outputs_bitwise_equal": True,
              "original_tensors_unchanged": True, "trainable_names": names,
              "trainable_parameters": 92160, "gradient_tensors": len(gradients),
              "nonzero_gradient_tensors": sum(bool(g.abs().max() > 0) for _, g in gradients),
              "full_objective_loss": float(loss.detach()), "amp_scale": AMP_INITIAL_SCALE,
              "adapter_backward_seconds": time.perf_counter()-backward,
              "total_seconds": time.perf_counter()-start, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
              "device": torch.cuda.get_device_name(), "actual_training_gpu_preflight_required": True,
              "pathology_improvement_established": False}
    (here / "local_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
