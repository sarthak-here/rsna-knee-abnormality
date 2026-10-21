"""CPU-only, weights-safe checkpoint inventory; never imports publisher source."""
import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch


def inventory(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("nonempty model state required")
    shapes, dtypes, families = {}, Counter(), Counter()
    elements = 0
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("model entries must be named tensors")
        if tensor.device.type != "cpu":
            raise ValueError("CPU tensors required")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"nonfinite tensor: {name}")
        shapes[name] = list(tensor.shape)
        dtypes[str(tensor.dtype)] += 1
        families[name.split(".")[0]] += 1
        elements += tensor.numel()
    # Deliberately inventory metadata names/types without publishing IDs/reports.
    metadata = {str(k): type(v).__name__ for k, v in payload.items() if k != "model"}
    result = {
        "root_keys": sorted(map(str, payload)),
        "metadata_types": metadata,
        "model_tensors": len(state),
        "model_elements": elements,
        "tensor_dtypes": dict(dtypes),
        "tensor_families": dict(families),
        "shape_identity_sha256": hashlib.sha256(
            json.dumps(shapes, sort_keys=True).encode()).hexdigest(),
        "all_model_tensors_finite": True,
        "metadata_is_independent_training_lineage_proof": False,
        "gpu_used": False,
        "publisher_source_executed": False,
        "predictions_generated": False,
    }
    contract = payload.get("contract")
    if isinstance(contract, Mapping):
        allowed = ("settings", "labels", "site_meta", "reference_folds", "ids",
                   "source_npz_hashes", "source", "weights", "upstream",
                   "python", "torch", "cuda", "numpy", "opencv", "gpu",
                   "policy", "precision", "fold")
        result["publisher_contract"] = {k: contract[k] for k in allowed if k in contract}
    if type(payload.get("epoch")) is int:
        result["publisher_epoch"] = payload["epoch"]
    score = payload.get("score")
    if isinstance(score, Mapping):
        result["publisher_score"] = {k: score[k] for k in
                                     ("macro", "per_target", "targets_scored") if k in score}
        result["publisher_score_is_verified_public_auc"] = False
    # Fail closed on non-JSON or nonfinite metadata, never pickle the report.
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    sha = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    # No unsafe-load retry, safe-global allowlisting or custom source imports.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    result = inventory(payload)
    result.update(checkpoint_sha256=sha.hexdigest(),
                  checkpoint_bytes=args.checkpoint.stat().st_size,
                  torch_version=torch.__version__)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
