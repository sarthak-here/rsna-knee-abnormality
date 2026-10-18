"""Build the exact private R6 stride-2 no-submission screen."""

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
    raise RuntimeError("cannot import multireader builder")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def patch_stride_two(source: str) -> str:
    old = "'--eval-stride', str(manifest.get('eval_stride', 3))"
    new = "'--eval-stride', '2'"
    if source.count(old) != 1:
        raise RuntimeError("R6 eval-stride patch anchor missing or ambiguous")
    source = source.replace(old, new, 1)
    marker = "print('ASSETS_READY', str(CODE_SRC), str(M_SRC), str(DATA), 'flip_tta', '--flip-tta' in WORKER_ARGS, 'seconds %.1f' % (time.time() - T0), flush=True)"
    replacement = "assert WORKER_ARGS[WORKER_ARGS.index('--eval-stride') + 1] == '2'\nprint('DENSE_WINDOW_CONTRACT eval_stride=2', flush=True)\n" + marker
    if source.count(marker) != 1:
        raise RuntimeError("R6 assets marker missing or ambiguous")
    return source.replace(marker, replacement, 1)


COLLECT = r'''from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

WORK = Path("/kaggle/working")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
EXPECTED_ANCHOR_SHA = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

truth = pd.read_csv(WORK / "gold_truth.csv", dtype={"StudyInstanceUID": str})
ids = truth.StudyInstanceUID.astype(str).tolist()
dense_frame = pd.read_csv(WORK / "r6_stride2_gold.csv", dtype={"StudyInstanceUID": str})
if dense_frame.columns.tolist() != ["StudyInstanceUID"] + TARGETS:
    raise RuntimeError("stride-2 schema mismatch")
if sorted(dense_frame.StudyInstanceUID) != sorted(ids) or dense_frame.StudyInstanceUID.nunique() != 58:
    raise RuntimeError("stride-2 ID mismatch")
dense = dense_frame.set_index("StudyInstanceUID").loc[ids, TARGETS].to_numpy(np.float32)
if not np.isfinite(dense).all() or (dense < 0).any() or (dense > 1).any():
    raise RuntimeError("invalid stride-2 probabilities")
receipt = json.loads((WORK / "r6_stride2_gold_receipt.json").read_text())
if receipt.get("status") != "PASS" or int(receipt.get("fallback_studies", -1)) != 0:
    raise RuntimeError("stride-2 R6 did not score all studies")

anchor_candidates = []
for path in Path("/kaggle/input").rglob("multireader_gold_predictions.npz"):
    if sha256(path) == EXPECTED_ANCHOR_SHA:
        anchor_candidates.append(path)
if len(anchor_candidates) != 1:
    raise RuntimeError(f"expected one exact anchor cache, found {anchor_candidates}")
anchor = anchor_candidates[0]

dense_path = WORK / "r6_dense_window_predictions.npz"
np.savez_compressed(
    dense_path,
    ids=np.asarray(ids, dtype="U80"),
    targets=np.asarray(TARGETS, dtype="U32"),
    predictions=dense,
)
(WORK / "r6_dense_window_receipt.json").write_text(json.dumps({
    "status": "PASS", "contract": "r6_eval_stride_2_replacement_v1",
    "source_notebook_sha256": "__R6_SHA__", "anchor_sha256": sha256(anchor),
    "dense_prediction_sha256": sha256(dense_path), "studies": 58,
    "targets": 12, "prediction_shape": [58, 12], "finite": True,
    "fallback_studies": 0, "r6_receipt": receipt,
    "submission_created": False,
}, indent=2) + "\n")
print("R6_DENSE_WINDOW_CACHE_COMPLETE", sha256(dense_path), flush=True)
'''


FINAL = r'''from pathlib import Path
import hashlib, json, subprocess, sys
WORK = Path("/kaggle/working")
evaluator = WORK / "evaluate_dense_window.py"
if hashlib.sha256(evaluator.read_bytes()).hexdigest() != "__EVALUATOR_SHA__":
    raise RuntimeError("evaluator identity mismatch")
anchor = None
for path in Path("/kaggle/input").rglob("multireader_gold_predictions.npz"):
    if hashlib.sha256(path.read_bytes()).hexdigest() == "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124":
        if anchor is not None:
            raise RuntimeError("duplicate exact anchor caches")
        anchor = path
if anchor is None:
    raise RuntimeError("exact anchor cache unavailable")
subprocess.run([
    sys.executable, str(evaluator), str(anchor),
    str(WORK / "r6_dense_window_predictions.npz"),
    "--out", str(WORK / "r6_dense_window_report.json"),
], check=True)
for path in WORK.rglob("*.csv"):
    path.unlink()
if list(WORK.rglob("*.csv")) or list(WORK.rglob("submission.csv")):
    raise RuntimeError("forbidden CSV artifact")
report = json.loads((WORK / "r6_dense_window_report.json").read_text())
receipt = json.loads((WORK / "r6_dense_window_receipt.json").read_text())
final = {
    "status": report["status"], "macro_delta": report["macro_delta"],
    "gates": report["gates"], "dense_prediction_sha256": receipt["dense_prediction_sha256"],
    "fallback_studies": receipt["fallback_studies"], "retained_csv_count": 0,
    "submission_created": False,
}
(WORK / "r6_dense_window_final_receipt.json").write_text(json.dumps(final, indent=2) + "\n")
print("R6_DENSE_WINDOW_NOTEBOOK_COMPLETE", json.dumps(final, indent=2), flush=True)
'''


def build(r6_path: Path, output: Path) -> None:
    if BASE.sha256(r6_path) != BASE.R6_SHA256:
        raise RuntimeError("R6 source notebook identity mismatch")
    source = json.loads(r6_path.read_text(encoding="utf-8"))
    evaluator_source = (HERE / "evaluate_dense_window.py").read_text(encoding="utf-8")
    evaluator_sha = hashlib.sha256(evaluator_source.encode()).hexdigest()
    stage = patch_stride_two(
        BASE.patch_r6_frozen_mount(BASE.patch_gold_root(BASE.cell_source(source, 6)))
    )
    collect = COLLECT.replace("__R6_SHA__", BASE.R6_SHA256)
    final = FINAL.replace("__EVALUATOR_SHA__", evaluator_sha)
    cells = [
        BASE.markdown("# RSNA knee R6 dense-window screen\n\nPrivate no-submission evaluation; exact public attribution is retained in the pinned R6 source.\n"),
        BASE.code(BASE.PREP),
        BASE.code(stage),
        BASE.code(BASE.cell_source(source, 7)),
        BASE.code(BASE.save_reader_cell("r6_stride2")),
        BASE.code(collect),
        BASE.code("%%writefile /kaggle/working/evaluate_dense_window.py\n" + evaluator_source),
        BASE.code(final),
    ]
    notebook = {
        "cells": cells,
        "metadata": source.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6", type=Path, default=BASE.DEFAULT_R6)
    parser.add_argument("--output", type=Path, default=HERE / "r6-dense-window-screen.ipynb")
    args = parser.parse_args()
    build(args.r6, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
