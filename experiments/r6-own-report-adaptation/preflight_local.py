"""GPU graph check, no optimizer: synthetic split-prefix parity and backward."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F


def main():
    repo = Path(__file__).resolve().parents[2]
    source = repo / ".audit/rsna-knee-blend-code/r6src"
    sys.path[:0] = [str(source), str(repo / ".audit/frontier-2026-10-17/goodpjw958/output/_own_env")]
    import compact_train as trainer
    from train import prefix, tail
    torch.set_num_threads(2); torch.manual_seed(20261020)
    assets = repo / ".audit/r6-strict-load"
    spec = json.loads((assets / "state_spec.json").read_text())
    checkpoint = assets / "E_lrprior_s0_f0.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != "59cc88a197bc80c7fa5faa89e69ae7bec04f69cab5259a23a0d0ceb9431e53f6":
        raise ValueError("checkpoint drift")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = trainer.Net("convnext_small", None, crop=352, state_spec=spec, in_chans=3, spatial=True, extent=True, syn_decomp=True)
    model.load_state_dict(state, strict=True)
    model = model.eval().requires_grad_(False).cuda()
    # Synthetic windows only: no gold, confirmation or test-study fitting.
    images = torch.randint(0, 256, (32, 3, 352, 352), dtype=torch.uint8, device="cuda")
    meta = torch.zeros((32, 9), device="cuda"); meta[:, 0] = 0; meta[:, 1] = 1
    meta[:, 3] = 1.5; meta[:, 4] = 1.; meta[:, 5] = 1; meta[:, 8] = 32
    mask = torch.ones((1, 32), dtype=torch.bool, device="cuda")
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        direct = model(images[None], meta[None], mask, chunk=8, return_aux=True)
        early = torch.cat([prefix(model, images[i:i+8]) for i in range(0, 32, 8)]).half()
        cached = tail(model, early, meta)
    valid = model.state_valid
    errors = [float((direct[0]-cached[0]).abs().max()),
              float((direct[1][:, valid]-cached[1][:, valid]).abs().max()),
              float((direct[2]-cached[2]).abs().max())]
    if max(errors) > .02:
        raise ValueError("cached-prefix graph differs from full reader")
    candidate = copy.deepcopy(model)
    candidate.enc.stages[3].requires_grad_(True).train()
    start_backward = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        z, sl, bl = tail(candidate, early.detach(), meta)
    loss = F.cross_entropy(sl[0, :4].float(), torch.zeros(4, device="cuda", dtype=torch.long))
    loss.backward(); torch.cuda.synchronize()
    grads = [(n, p.grad) for n, p in candidate.named_parameters() if p.grad is not None]
    if len(grads) != 31 or any(not n.startswith("enc.stages.3.") or not torch.isfinite(g).all() for n, g in grads):
        raise ValueError("incorrect/nonfinite gradient scope")
    result = {"checkpoint_sha256": digest, "device": torch.cuda.get_device_name(),
              "prefix_shape": list(early.shape), "prefix_dtype": str(early.dtype),
              "max_full_vs_cached_logit_errors": errors, "gradient_tensors": len(grads),
              "nonzero_gradient_tensors": sum(bool(g.abs().max() > 0) for _, g in grads),
              "last_stage_backward_seconds": time.perf_counter()-start_backward,
              "total_seconds": time.perf_counter()-start, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
              "optimizer_steps": 0, "synthetic_only": True, "pathology_score": None,
              "kaggle_measurement_required": True,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(__file__).with_name("local_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
