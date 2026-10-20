"""Actual official preview DICOMs: full vs lean native decode/quantizer parity."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--probe-worker", required=True)
    parser.add_argument("--deps", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    pins = {Path(args.code_root) / "training/native_cache.py": "c53ce8985406b3b918d9e821d43518479e89d9f946daebd817f378e3949718b9",
            Path(args.code_root) / "training/dicom_geometry.py": "12aa501f9704aef15b7a3bbd560951acb6d95b89b7de0141d6cf89fc3f051ab1",
            Path(args.probe_worker): "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980"}
    for path, expected_sha in pins.items():
        if sha(path) != expected_sha:
            raise ValueError(f"native source identity drift: {path}")
    sys.path.insert(0, str(Path(args.deps).resolve()))
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    import pydicom
    if pydicom.__version__ != "3.0.1":
        raise ValueError("review DICOM-reader version before changing runtime")
    from training import native_cache as native
    from training import dicom_geometry as geom
    spec = importlib.util.spec_from_file_location("native_parity_probe_worker", args.probe_worker)
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    root = Path(args.fixture)
    receipt = json.loads((root / "preview_fixture_receipt.json").read_text())
    paths = [root / r["path"] for r in receipt["files"]]
    if len(paths) < 3 or receipt["bytes"] > 96 * 1024 ** 2:
        raise ValueError("invalid bounded fixture")
    for path, record in zip(paths, receipt["files"]):
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("raw DICOM identity mismatch")
    sid = paths[0].parent.name
    metadata = pd.read_csv(root / "test_series.csv", dtype=str)
    series = metadata.loc[metadata.SeriesInstanceUID == sid]
    if len(series) != 1:
        raise ValueError("series metadata identity mismatch")
    plane = series.iloc[0].Anatomical_Plane
    full = {str(p): pydicom.dcmread(p) for p in paths}
    lean = {str(p): pydicom.dcmread(p, stop_before_pixels=True) for p in paths}
    full_order = geom.order_series(Path(sid), plane, provided_records=[geom.header_from_dataset(ds, p) for p, ds in full.items()])
    lean_order = geom.order_series(Path(sid), plane, provided_records=[geom.header_from_dataset(ds, p) for p, ds in lean.items()])
    if full_order.files != lean_order.files or not np.array_equal(full_order.positions_mm, lean_order.positions_mm):
        raise ValueError("lean header read changes physical ordering")
    if len(full_order.files) != len(paths):
        raise ValueError("source ordering silently omitted raw slices")
    images, lean_images = [], []
    transfer_syntaxes = set()
    for path in full_order.files:
        decoded = geom.canonicalize_payload(geom.pixels_from_dataset(full[path], path=path), plane)
        reread = geom.canonicalize_payload(geom.pixels_from_dataset(pydicom.dcmread(path), path=path), plane)
        if not np.array_equal(decoded.image, reread.image) or not np.array_equal(decoded.spacing, reread.spacing):
            raise ValueError("full vs lean reread pixels or spacing mismatch")
        if not np.isfinite(decoded.image).all():
            raise ValueError("nonfinite decoded pixels")
        images.append(decoded.image); lean_images.append(reread.image)
        transfer_syntaxes.add(str(full[path].file_meta.TransferSyntaxUID))
    original, original_info = native.quantize_series(images)
    runtime, runtime_info = worker.quantize_series_slicewise(lean_images)
    if original.dtype != np.uint16 or not np.array_equal(original, runtime) or original_info != runtime_info:
        raise ValueError("native cache vs inference quantizer differs on actual pixels")
    result = {"scope": "real preview single-series pre-laterality native decoder/quantizer parity only",
              "fixture_receipt_sha256": sha(root / "preview_fixture_receipt.json"),
              "dicom_files": len(paths), "raw_bytes": receipt["bytes"], "plane": plane,
              "native_shape": list(original.shape), "native_dtype": str(original.dtype),
              "physical_header_order_identical": True, "full_vs_lean_canonical_pixels_and_spacing_identical": True,
              "uint16_quantization_bitwise_equal": True, "display_window_and_quantization_metadata_exact": True,
              "native_array_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
              "cpu_audit_seconds_excluding_python_startup": time.perf_counter() - started,
              "pydicom_version": pydicom.__version__, "transfer_syntaxes": sorted(transfer_syntaxes),
              "source_hashes": {"native_cache.py": sha(native.__file__), "dicom_geometry.py": sha(geom.__file__),
                                "probe_infer.py": sha(args.probe_worker), "audit_script": sha(__file__)},
              "study_side_resolution_tested": False, "anchor_inference_tested": False,
              "anatomy_centred_v4_render_tested": False, "pixel_derived_fluid_volume_tested": False,
              "scope_is_complete_study": False, "gpu_used": False, "optimizer_steps": 0,
              "study_predictions": 0, "training_labels_used": False, "promotion": False}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
