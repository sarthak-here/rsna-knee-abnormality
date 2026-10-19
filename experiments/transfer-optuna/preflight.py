"""Bounded synthetic GPU execution check. Not a scientific validation."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from models import Reader, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    assets = json.loads(Path(args.assets).read_text())
    rows = []
    for family in ("swin3d_t", "videomae"):
        model = None
        try:
            torch.manual_seed(20261019)
            model = Reader(family, assets).cuda().train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
            x = torch.zeros(1, 6, 3, 16, 224, 224, device="cuda")
            mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(3):
                torch.cuda.synchronize(); start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                # Full precision makes this a conservative execution check.
                prediction = model(x, mask)
                loss = prediction.float().square().mean()
                loss.backward()
                if not torch.isfinite(loss) or not all(
                    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
                ):
                    raise ValueError("nonfinite synthetic gradient")
                optimizer.step()
                torch.cuda.synchronize(); times.append(time.monotonic() - start)
            rows.append({"family": family, "passed": True, "step_seconds": times,
                         "peak_bytes": torch.cuda.max_memory_allocated(),
                         "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        except Exception as error:
            rows.append({"family": family, "passed": False,
                         "error_type": type(error).__name__, "error": str(error)})
        finally:
            if model is not None:
                del model
            # Optimizer may own model tensors after a failed step.
            if "optimizer" in locals():
                del optimizer
            gc.collect(); torch.cuda.empty_cache()
    report = {"synthetic_only": True, "predictive_trials": 0,
              "models_source_sha256": sha256(Path(__file__).with_name("models.py")),
              "weight_hashes": {key: assets[key]["sha256"] for key in ("swin3d_t", "videomae")},
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "models": rows}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(row["passed"] for row in rows):
        raise SystemExit("Preflight failed: do not launch this two-model search")


if __name__ == "__main__":
    main()
