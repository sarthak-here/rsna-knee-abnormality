"""CPU-only R6 reconstruction and final-stage parameter selection; no optimizer."""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

CHECKPOINT_SHA = "59cc88a197bc80c7fa5faa89e69ae7bec04f69cab5259a23a0d0ceb9431e53f6"
SPEC_SHA = "3914242b0c404aee625ea65801f14f60adfee326c1ddd6fb96c48ed7050b7ff3"
TRAINER_SHA = "04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(path, expected):
    if sha(path) != expected:
        raise ValueError(f"input identity drift: {path}")


def last_stage_only(model):
    model.requires_grad_(False)
    if not hasattr(model.enc, "stages") or len(model.enc.stages) != 4:
        raise ValueError("unexpected ConvNeXt stage layout")
    model.enc.stages[3].requires_grad_(True)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not names or not all(name.startswith("enc.stages.3.") for name in names):
        raise ValueError("incorrect trainable scope")
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-spec", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--timm-root", required=True, help="directory containing the audited timm package")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    require(args.checkpoint, CHECKPOINT_SHA); require(args.state_spec, SPEC_SHA); require(args.trainer, TRAINER_SHA)
    sys.path.insert(0, str(Path(args.timm_root).resolve()))
    import timm
    if timm.__version__ != "1.0.25":
        raise ValueError("review timm implementation before changing the pinned runtime")
    torch.set_num_threads(4)
    module_spec = importlib.util.spec_from_file_location("r6_original_trainer", args.trainer)
    trainer = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(trainer)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(t, torch.Tensor) for t in state.values()):
        raise ValueError("expected plain whole-model tensor state")
    assert all(torch.isfinite(t).all().item() for t in state.values() if t.is_floating_point())
    state_spec = json.loads(Path(args.state_spec).read_text())
    # Exactly the flag inference used by the released inference worker.
    flags = {"spatial": "sp_q" in state, "extent": any(k in state for k in ("ext_u", "ext_view", "ext_gamma")),
             "syn_decomp": "syn_beta" in state}
    reference = trainer.Net("convnext_small", None, crop=352, state_spec=state_spec, in_chans=3, **flags)
    reference.load_state_dict(state, strict=True)
    reference.eval().requires_grad_(False)
    if set(reference.state_dict()) != set(state):
        raise ValueError("strict state identity failure")
    assert all(torch.equal(reference.state_dict()[k], v) for k, v in state.items())
    assert all(reference.state_dict()[k].dtype == v.dtype for k, v in state.items())
    # A missing head tensor must fail; no strict=False/backbone-only escape.
    broken = dict(state); broken.pop("cls_w")
    try:
        reference.load_state_dict(broken, strict=True)
    except RuntimeError:
        negative_load_rejected = True
    else:
        raise ValueError("missing head tensor was accepted")
    reference.load_state_dict(state, strict=True)
    candidate = copy.deepcopy(reference)
    trainable = last_stage_only(candidate)
    assert all(torch.equal(candidate.state_dict()[k], v) for k, v in state.items())
    # This synthetic head-only check is not raw-image/preprocessing equivalence.
    generator = torch.Generator().manual_seed(20261020)
    features = torch.randn(2, 768, 11, 11, generator=generator) if flags["spatial"] else torch.randn(2, 768, generator=generator)
    meta = torch.zeros(1, 2, 9)
    meta[0, :, 0] = 0; meta[0, :, 1] = torch.tensor([0, 4])
    meta[..., 3] = 1.; meta[..., 8] = 2.
    mask = torch.ones(1, 2, dtype=torch.bool)
    with torch.inference_mode():
        def output(model):
            tokens = model.spatial_pool(features) if flags["spatial"] else features
            return model.head(tokens, meta, mask, return_aux=True)
        before, after = output(reference), output(candidate)
    for a, b in zip(before, after):
        assert torch.equal(a, b), "freezing changed initial outputs"
    assert before[0].shape == (1, 12) and torch.isfinite(before[0]).all()
    assert torch.isfinite(before[1][:, reference.state_valid]).all()
    assert torch.isfinite(before[2]).all()
    state_probs = before[1].softmax(-1)
    assert torch.isfinite(state_probs).all() and torch.allclose(state_probs.sum(-1), torch.ones(1, 12))
    timm_dir = Path(timm.__file__).resolve().parent
    report = {"checkpoint_sha256": CHECKPOINT_SHA, "state_spec_sha256": SPEC_SHA,
              "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
              "trainer_sha256": TRAINER_SHA, "timm_version": timm.__version__,
              "timm_source_hashes": {name: sha(timm_dir / name) for name in
                                     ("__init__.py", "models/convnext.py", "models/_factory.py")},
              "torch_version": torch.__version__, "device": "cpu", "checkpoint_tensors": len(state),
              "all_checkpoint_tensors_exact_after_load": True, "all_tensor_dtypes_preserved": True, "flags": flags,
              "missing_head_negative_test_rejected": negative_load_rejected,
              "candidate_state_unchanged_at_initialization": True,
              "synthetic_head_outputs_bitwise_equal": True, "state_and_binary_logits_checked": True,
              "trainable_parameter_tensors": len(trainable),
              "trainable_parameters": sum(p.numel() for p in candidate.parameters() if p.requires_grad),
              "total_parameters": sum(p.numel() for p in candidate.parameters()),
              "trainable_names": trainable, "frozen_heads": True,
              "gpu_used": False, "optimizer_steps": 0, "study_predictions": 0,
              "raw_pixel_preprocessing_parity_tested": False, "postprocessor_parity_tested": False,
              "promotion": False}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trainable_names"}, indent=2))


if __name__ == "__main__":
    main()
