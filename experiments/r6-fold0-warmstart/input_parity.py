"""Real-window crop and normalization audit; stop before encoder computation."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace


def check(images, metadata, trainer, source, assets, crop):
    import torch
    import json
    # Select actual views/slots, without labels, rather than synthetic pixels.
    selected = []
    seen = set()
    for i, row in enumerate(metadata):
        key = (int(row[0]), int(row[1]))
        if key not in seen:
            seen.add(key)
            selected.append(i)
    full = images[selected]
    a = SimpleNamespace(crop=crop)
    train_tree = ast.parse((source / "compact_train.py").read_text(encoding="utf-8"))
    functions = [n for n in ast.walk(train_tree) if isinstance(n, ast.FunctionDef) and n.name == "center"]
    if len(functions) != 1:
        raise ValueError("ambiguous original training centre crop")
    node = copy.deepcopy(functions[0]); node.decorator_list = []
    context = {"a": a}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "source_training_center", "exec"), context)
    training = context["center"](full[None])
    worker_tree = ast.parse((source / "compact_infer.py").read_text(encoding="utf-8"))
    assignments = [n for n in ast.walk(worker_tree) if isinstance(n, ast.Assign) and
                   ast.unparse(n) == "img = full[None, :, :, off:off + a.crop, off:off + a.crop]"]
    if len(assignments) != 1:
        raise ValueError("review original inference crop before continuing")
    context = {"a": a, "off": (full.shape[-1] - crop) // 2, "full": full}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(assignments[0])], type_ignores=[])),
                 "source_worker_center", "exec"), context)
    inference = context["img"]
    if not torch.equal(training, inference):
        raise ValueError("source training/inference centre crops differ on real windows")
    checkpoint = assets / "E_lrprior_s0_f0.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != "59cc88a197bc80c7fa5faa89e69ae7bec04f69cab5259a23a0d0ceb9431e53f6":
        raise ValueError("fold-0 checkpoint identity mismatch")
    spec_path = assets / "state_spec.json"
    if hashlib.sha256(spec_path.read_bytes()).hexdigest() != "3914242b0c404aee625ea65801f14f60adfee326c1ddd6fb96c48ed7050b7ff3":
        raise ValueError("state specification identity mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = trainer.Net("convnext_small", None, crop=crop, state_spec=json.loads(spec_path.read_text()), in_chans=3,
                        spatial="sp_q" in state, extent="ext_u" in state, syn_decomp="syn_beta" in state)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    mask = torch.ones((1, len(full)), dtype=torch.bool)
    captured = {}

    class BeforeEncoder(Exception):
        pass

    def capture(module, args):
        captured["input"] = args[0].clone()
        raise BeforeEncoder()

    handle = model.enc.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            model(inference, metadata[selected][None], mask, chunk=32)
    except BeforeEncoder:
        pass
    finally:
        handle.remove()
    if "input" not in captured:
        raise ValueError("normalization interception did not reach encoder boundary")
    pixels = training[mask]
    if len(pixels) % 32:
        pixels = torch.cat([pixels, pixels.new_zeros((32 - len(pixels) % 32,) + tuple(pixels.shape[1:]))])
    expected = ((pixels[:32].float() / 255. - model.mean) / model.std).contiguous(memory_format=torch.channels_last)
    got = captured["input"]
    if not torch.equal(expected, got) or not torch.isfinite(got).all():
        raise ValueError("encoder-boundary pixels differ from preserved normalization")
    return {"sampled_real_views": len(full), "slot_view_pairs": sorted([list(x) for x in seen]),
            "source_crop_functions_bitwise_equal": True, "cropped_shape": list(training.shape),
            "encoder_boundary_shape": list(got.shape), "encoder_normalization_bitwise_equal": True,
            "normalized_input_sha256": hashlib.sha256(got.numpy().tobytes()).hexdigest(),
            "normalization_mean": model.mean.flatten().tolist(), "normalization_std": model.std.flatten().tolist(),
            "channels_last": got.is_contiguous(memory_format=torch.channels_last),
            "encoder_forward_executed": False, "disease_head_executed": False, "optimizer_steps": 0,
            "scope": "real sampled views and padded windows; intercepted before any encoder computation",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
