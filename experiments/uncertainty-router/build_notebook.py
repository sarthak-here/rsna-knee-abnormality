"""Build the exact no-submission 58-study uncertainty-router notebook."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / "multireader-nested-blend" / "build_notebook.py"
SPEC = importlib.util.spec_from_file_location("multireader_builder", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load multireader builder")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def patch_r6_logit_dump(source: str) -> str:
    marker = "print('ASSETS_READY', str(CODE_SRC), str(M_SRC), str(DATA), 'flip_tta', '--flip-tta' in WORKER_ARGS, 'seconds %.1f' % (time.time() - T0), flush=True)"
    if source.count(marker) != 1:
        raise RuntimeError("R6 worker argument marker missing or ambiguous")
    addition = (
        "WORKER_ARGS += ['--dump-logits', '/kaggle/working/r6_fold_logits.json']\n"
        + marker
    )
    source = source.replace(marker, addition, 1)
    old_dump = '    if a.dump_logits:\\n        Path(a.dump_logits).write_text(json.dumps(dump), encoding="utf-8")'
    new_dump = (
        '    if a.dump_logits:\\n'
        '        _dump_path = Path(a.dump_logits)\\n'
        '        _gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "single")\\n'
        '        _dump_path = _dump_path.with_name(_dump_path.stem + "_worker" + _gpu_tag + _dump_path.suffix)\\n'
        '        _dump_path.write_text(json.dumps(dump), encoding="utf-8")'
    )
    if source.count(old_dump) != 1:
        raise RuntimeError("R6 dump writer patch anchor missing or ambiguous")
    return source.replace(old_dump, new_dump, 1)


COLLECT = r'''from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

WORK = Path("/kaggle/working")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
PERM = np.asarray([0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11])

def frame(path, ids):
    value = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if value.columns.tolist() != ["StudyInstanceUID"] + TARGETS:
        raise RuntimeError(f"schema mismatch: {path}")
    if sorted(value.StudyInstanceUID) != sorted(ids) or value.StudyInstanceUID.nunique() != 58:
        raise RuntimeError(f"ID mismatch: {path}")
    out = value.set_index("StudyInstanceUID").loc[ids, TARGETS].to_numpy(np.float64)
    if not np.isfinite(out).all() or (out < 0).any() or (out > 1).any():
        raise RuntimeError(f"invalid probability: {path}")
    return out

truth_frame = pd.read_csv(WORK / "gold_truth.csv", dtype={"StudyInstanceUID": str})
ids = truth_frame.StudyInstanceUID.astype(str).tolist()
if len(ids) != 58 or len(set(ids)) != 58:
    raise RuntimeError("gold ID drift")

r6_prob = frame(WORK / "r6_flip_gold.csv", ids)
r384 = frame(WORK / "raptor_raw.csv", ids)
r224 = frame(WORK / "raptor224_raw.csv", ids)

parts = sorted(WORK.glob("r6_fold_logits_worker*.json"))
if len(parts) != 2:
    raise RuntimeError(f"expected two worker logit dumps, found {parts}")
raw = {}
for path in parts:
    part = json.loads(path.read_text())
    overlap = set(raw).intersection(part)
    if overlap:
        raise RuntimeError(f"duplicate IDs across worker logit dumps: {sorted(overlap)[:3]}")
    raw.update(part)
if set(raw) != set(ids):
    raise RuntimeError("R6 logit dump ID mismatch")
r6_logits = np.empty((58, 4, 2, 12), np.float32)
for i, uid in enumerate(ids):
    rec = raw[uid]
    plain = np.asarray(rec.get("z_plain"), np.float32)
    mirror_raw = np.asarray(rec.get("z_mirror_raw"), np.float32)
    if plain.shape != (4, 12) or mirror_raw.shape != (4, 12):
        raise RuntimeError(f"R6 fold/pass shape mismatch for {uid}: {plain.shape}, {mirror_raw.shape}")
    mirror = mirror_raw[:, PERM]
    mirror[:, 1] = plain[:, 1]  # MCL is explicitly plain-pass-only in the frozen worker.
    r6_logits[i, :, 0] = plain
    r6_logits[i, :, 1] = mirror

receipts = {
    "r6": json.loads((WORK / "r6_flip_gold_receipt.json").read_text()),
    "raptor384": json.loads((WORK / "raptor384_gold_receipt.json").read_text()),
    "raptor224": json.loads((WORK / "raptor224_gold_receipt.json").read_text()),
}
if receipts["r6"].get("status") != "PASS" or receipts["r6"].get("fallback_studies") != 0:
    raise RuntimeError("R6 receipt failed")
for name in ("raptor384", "raptor224"):
    receipt = receipts[name]
    if not receipt.get("complete") or receipt.get("preprocess_failures") != 0 or receipt.get("inference_failures") != 0:
        raise RuntimeError(f"{name} receipt failed")
if not np.isfinite(r6_logits).all():
    raise RuntimeError("non-finite R6 fold logits")

cache = WORK / "uncertainty_router_gold_cache.npz"
np.savez_compressed(
    cache,
    ids=np.asarray(ids, dtype="U80"),
    truth=truth_frame[TARGETS].to_numpy(np.int8),
    r6_prob=r6_prob.astype(np.float32),
    r6_logits=r6_logits,
    raptor_prob=np.stack([r384, r224], axis=1).astype(np.float32),
)
cache_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
(WORK / "uncertainty_router_cache_receipt.json").write_text(json.dumps({
    "status": "PASS", "studies": 58, "targets": 12,
    "r6_probability_shape": [58, 12], "r6_logit_shape": [58, 4, 2, 12],
    "raptor_probability_shape": [58, 2, 12], "finite": True,
    "cache_sha256": cache_sha, "receipts": receipts, "submission_created": False,
}, indent=2) + "\n")
print("UNCERTAINTY_CACHE_COMPLETE", cache_sha, flush=True)
'''


FINAL = r'''from pathlib import Path
import hashlib, json, subprocess, sys
WORK = Path("/kaggle/working")
evaluator = WORK / "evaluate_uncertainty_router.py"
expected = "__EVALUATOR_SHA__"
actual = hashlib.sha256(evaluator.read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError("evaluator identity mismatch")
subprocess.run([
    sys.executable, str(evaluator), str(WORK / "uncertainty_router_gold_cache.npz"),
    "--out", str(WORK / "uncertainty_router_report.json"),
], check=True)
report = json.loads((WORK / "uncertainty_router_report.json").read_text())
receipt = json.loads((WORK / "uncertainty_router_cache_receipt.json").read_text())
for path in WORK.rglob("*.csv"):
    path.unlink()
if list(WORK.rglob("submission.csv")) or list(WORK.rglob("*.csv")):
    raise RuntimeError("forbidden CSV artifact")
final = {
    "status": "PASS" if report["passed"] else "REJECTED",
    "source_notebook_sha256": "__R6_SHA__",
    "evaluator_sha256": actual,
    "cache_sha256": receipt["cache_sha256"],
    "macro_delta": report["macro_delta"], "gates": report["gates"],
    "submission_created": False, "retained_csv_count": 0,
}
(WORK / "uncertainty_router_final_receipt.json").write_text(json.dumps(final, indent=2) + "\n")
print("UNCERTAINTY_ROUTER_NOTEBOOK_COMPLETE", json.dumps(final, indent=2), flush=True)
'''


def build(r6_path: Path, output: Path) -> None:
    if BASE.sha256(r6_path) != BASE.R6_SHA256:
        raise RuntimeError("R6 source notebook identity mismatch")
    r6 = json.loads(r6_path.read_text(encoding="utf-8"))
    evaluator_source = (HERE / "evaluate_uncertainty_router.py").read_text(encoding="utf-8")
    evaluator_sha = hashlib.sha256(evaluator_source.encode()).hexdigest()
    r6_stage = patch_r6_logit_dump(
        BASE.patch_r6_frozen_mount(BASE.patch_gold_root(BASE.cell_source(r6, 6)))
    )
    final = (
        FINAL.replace("__EVALUATOR_SHA__", evaluator_sha)
        .replace("__R6_SHA__", BASE.R6_SHA256)
    )
    cells = [
        BASE.markdown("# RSNA knee fold-uncertainty router\n\nPrivate no-submission evaluation. The Raptor checkpoints are acknowledged as OAI-trained.\n"),
        BASE.code(BASE.PREP),
        BASE.code(BASE.patch_raptor_receipt(BASE.cell_source(r6, 2), "raptor384")),
        BASE.code(BASE.cell_source(r6, 3)),
        BASE.code(BASE.patch_raptor_receipt(BASE.cell_source(r6, 4), "raptor224")),
        BASE.code(BASE.cell_source(r6, 5)),
        BASE.code(r6_stage),
        BASE.code(BASE.cell_source(r6, 7)),
        BASE.code(BASE.save_reader_cell("r6_flip")),
        BASE.code(COLLECT),
        BASE.code("%%writefile /kaggle/working/evaluate_uncertainty_router.py\n" + evaluator_source),
        BASE.code(final),
    ]
    notebook = {
        "cells": cells,
        "metadata": r6.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6", type=Path, default=BASE.DEFAULT_R6)
    parser.add_argument("--output", type=Path, default=HERE / "uncertainty-router.ipynb")
    args = parser.parse_args()
    build(args.r6, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
