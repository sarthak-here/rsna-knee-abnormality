"""CPU-only real preview: deployed renderer versus original training eval loader.

No disease network, optimizer, labels, coefficient fitting or submission is run.
The preview is a transport/implementation fixture, not independent validation.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # before importing any torch-dependent source
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--fixture-receipt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-model-input", action="store_true", help="Intercept real normalized inputs before encoder computation")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = repo / ".audit/rsna-knee-blend-code/r6src"
    code = repo / ".audit/rsna-knee-blend-code/cloud_code"
    assets = repo / ".audit/r6-strict-load"
    pins = {
        source / "compact_infer.py": "8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d",
        source / "compact_train.py": "04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba",
        source / "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
        source / "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309",
        source / "side_common.py": "bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566",
        repo / ".audit/r6-w1/manifest.json": "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648",
        assets / "anatomy.pt": "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e",
        assets / "side_cls.pt": "ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a",
    }
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError(f"pinned source/asset mismatch: {path.name}")
    config = json.loads((repo / ".audit/r6-w1/manifest.json").read_text())
    stride, physical_step = config["eval_stride"], config["phys_mm"]
    fixture = Path(args.fixture).resolve()
    manifest_path = Path(args.fixture_receipt).resolve()
    if manifest_path.parent != fixture:
        raise ValueError("fixture receipt must belong to this fixture root")
    receipt = json.loads(manifest_path.read_text())
    uid = receipt["study_uid"]
    if receipt["split"] != "public test preview" or receipt["bytes"] > 512 * 1024 ** 2:
        raise ValueError("not a bounded official preview study")
    for record in receipt["files"]:
        path = (fixture / record["path"]).resolve()
        if fixture not in path.parents or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("fixture raw DICOM identity mismatch")
    sys.path[:0] = [str(repo / ".audit/r6-cpu-deps"), str(source), str(code),
                   str(repo / ".audit/frontier-2026-10-17/goodpjw958/output/_own_env")]
    import torch
    import numpy as np
    import pandas as pd
    assert not torch.cuda.is_available(), "CPU-only gate must not see CUDA"
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("preview_v4_worker", source / "compact_infer.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    import compact_train as trainer
    metadata = pd.read_csv(fixture / "test_series.csv", dtype=str)
    expected_series = set(metadata.loc[metadata.StudyInstanceUID == uid, "SeriesInstanceUID"])
    if expected_series != {Path(r["path"]).parent.name for r in receipt["files"]}:
        raise ValueError("fixture does not cover all official study series")
    tmp = Path(tempfile.mkdtemp(prefix="r6v4-", dir=repo / ".audit")).resolve()
    # The unmodified upstream worker removes only this exclusive per-study child.
    expected_work = tmp / ("s_" + hashlib.sha256(uid.encode()).hexdigest()[:16])
    assert expected_work.resolve().parent == tmp and tmp.parent == repo / ".audit"
    ds = worker.CStudies(fixture, "test", [uid], tmp, code, 384, eval_stride=stride,
                         views="v4", slices=3, anchor_weights=assets / "anatomy.pt",
                         phys_mm=physical_step, side_model=assets / "side_cls.pt", extent_meta=True, fluid=True)
    started = time.perf_counter()
    ds._setup()
    captured = {}
    original_build = ds.builder.build
    original_record = ds.bank.record

    def capture_build(*a, **kw):
        item = original_build(*a, **kw)
        captured["item"] = item
        return item

    def capture_record(*a, **kw):
        rec = original_record(*a, **kw)
        captured["anchor"] = rec
        return rec

    ds.builder.build = capture_build
    ds.bank.record = capture_record
    out = ds[0]
    if out["error"] is not None:
        raise ValueError("upstream worker failed closed: " + out["error"])
    item = captured["item"]
    anchor = captured["anchor"]
    if {b["series"] for b in item["blocks"]} != expected_series or item["skipped_series"]:
        raise ValueError("renderer omitted or skipped an official study series")
    if any(str(x).startswith("error:") for x in anchor["why"]):
        raise ValueError("anchor model/runtime error; do not silently accept its geometry fallback")
    segmented_anchor = str(anchor["method"]).startswith("skm_tea_joint")
    if not segmented_anchor and anchor["method"] != "slab_prior":
        raise ValueError("no usable published anchor path")
    if any(not e.get("in_field") for e in anchor["series"].values()) or set(anchor["series"]) != expected_series:
        raise ValueError("missing/out-of-field anchor would allow a field-centre substitute")
    cache = tmp / "training_eval_cache"
    cache.mkdir()
    np.save(cache / "px.u8.npy", item["px"].numpy(), allow_pickle=False)
    np.save(cache / "meta.f32.npy", item["meta"].numpy(), allow_pickle=False)
    (cache / "index.json").write_text(json.dumps({"blocks": {uid: item["blocks"]}}))
    train_eval = trainer.Studies(str(cache), [uid], 176, "eval", 0, eval_stride=stride,
                                 slices=3, phys_mm=physical_step, extent_meta=True)
    images, meta, _ = train_eval[0]
    if not torch.equal(images, out["images"]) or not torch.equal(meta, out["meta"]):
        raise ValueError("original training eval windows/meta differ from deployed loader")
    if not torch.isfinite(meta).all() or images.dtype != torch.uint8:
        raise ValueError("invalid real window tensors")
    fluid = out["fluid"]
    if str(fluid["status"]).startswith("error:"):
        raise ValueError("pixel fluid code failed: " + fluid["status"])
    if fluid["status"] == "ok" and not all(fluid.get(k) is not None and np.isfinite(fluid[k]) and fluid[k] >= 0
                                           for k in ("eff_sag_p99_open", "bak_sag_p99_open")):
        raise ValueError("invalid pixel-derived fluid measurement")
    input_parity = None
    if args.check_model_input:
        from input_parity import check
        input_parity = check(images, meta, trainer, source, assets, config["crop"])
    report = {
        "scope": "one complete official preview study; original training eval vs deployed CPU loader",
        "fixture_receipt_sha256": sha(manifest_path), "dicom_count": len(receipt["files"]),
        "series_count": len(expected_series), "raw_bytes": receipt["bytes"],
        "assets": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in pins},
        "anchor": anchor, "side": out["side"], "images_shape": list(images.shape),
        "segmentation_derived_anchor_accepted": segmented_anchor,
        "explicit_source_geometry_prior_used": anchor["method"] == "slab_prior",
        "images_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
        "metadata_shape": list(meta.shape), "metadata_sha256": hashlib.sha256(meta.numpy().tobytes()).hexdigest(),
        "training_eval_and_runtime_windows_bitwise_equal": True,
        "training_eval_and_runtime_metadata_bitwise_equal": True,
        "settings": {"size": 384, "crop": 352, "views": "v4", "stride": stride,
                     "through_plane_step_mm": physical_step, "local_crop_mm": 100., "slices": 3},
        "fluid": fluid, "fluid_measurement_branch_exercised": fluid["status"] == "ok",
        "positive_fluid_measurement_exercised": fluid["status"] == "ok" and any(
            fluid[k] > 0 for k in ("eff_sag_p99_open", "bak_sag_p99_open")),
        "disease_network_forward_tested": False, "encoder_normalization_parity_tested": input_parity is not None,
        "model_input_parity": input_parity,
        "gpu_used": False, "optimizer_steps": 0, "pathology_predictions": 0,
        "labels_used": False, "submission_created": False, "advancement_authorized": False,
        "cpu_seconds_including_aux_model_load": time.perf_counter() - started,
        "limits": "Single preview fixture, no proof of every protocol or training augmentation; no accuracy evidence.",
        "audit_script_sha256": sha(__file__),
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
