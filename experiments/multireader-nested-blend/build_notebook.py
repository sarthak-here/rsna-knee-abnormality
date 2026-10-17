"""Build the exact four-reader, gold-only prediction-cache notebook.

This builder composes the already-audited public R6 and R7 notebooks. It changes
only the evaluation root (58 labelled train studies instead of hidden test),
adds fail-closed receipts, retains each arm separately, runs the pinned nested
optimizer, and removes every CSV from /kaggle/working.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_R6 = Path(r"C:\Users\sarthak\Desktop\codex\audit-09576\0-9576-latest-public-dataset-score-in-one-click.ipynb")
DEFAULT_R7 = Path(r"C:\Users\sarthak\Desktop\codex\audit-gp958d\rsna-knee-gp958d.ipynb")
R6_SHA256 = "16a9d953972d9d82717f279f75e57e798d3902f93d1071c30f3caa67326e3601"
R7_SHA256 = "85c694c903ec6841488b49d5ecbe2838b8fad7db283f1f952c59f48d936e0b24"


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def cell_source(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def patch_gold_root(source: str) -> str:
    source = source.replace("ROOT = find_test_root()", 'ROOT = "/kaggle/working/gold_eval"')
    source = source.replace(
        "DATA = find_mount(COMPETITION, 'test.csv')",
        "DATA = Path('/kaggle/working/gold_eval')",
    )
    return source


def patch_r6_frozen_mount(source: str) -> str:
    """Point R6 at the byte-identical v1 mirror while R7 uses v2 of the original slug."""
    old = "C_SLUG = 'rsna-knee-r6-frozen-fe055406'"
    new = "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'"
    if source.count(old) != 1:
        raise RuntimeError("R6 frozen-mount patch anchor missing or ambiguous")
    return source.replace(old, new, 1)


def patch_raptor_receipt(source: str, stem: str) -> str:
    source = patch_gold_root(source)
    old_pre = '''def _pre_one(sid):
    try:
        vol, mask = _READER(sid, _SER, _TSDIR)
    except Exception as e:
        print(f"  [pre] {sid[:16]} FAILED ({type(e).__name__}: {e})", flush=True)
        vol = np.zeros((MAXS, IMG, IMG), np.uint8); mask = np.zeros(MAXS, np.uint8)
    return sid, vol, mask'''
    new_pre = '''def _pre_one(sid):
    ok = True
    try:
        vol, mask = _READER(sid, _SER, _TSDIR)
    except Exception as e:
        ok = False
        print(f"  [pre] {sid[:16]} FAILED ({type(e).__name__}: {e})", flush=True)
        vol = np.zeros((MAXS, IMG, IMG), np.uint8); mask = np.zeros(MAXS, np.uint8)
    return sid, vol, mask, ok'''
    if old_pre not in source:
        raise RuntimeError(f"{stem}: preprocessor patch anchor missing")
    source = source.replace(old_pre, new_pre)
    source = source.replace(
        "q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()",
        "q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()\n"
        "    pre_failed = set(); infer_failed = set()",
    )
    source = source.replace(
        'print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)',
        'infer_failed.add(sid)\n'
        '                print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)',
    )
    old_loop = "for i, (sid, vol, mask) in enumerate(pool.imap_unordered(_pre_one, test_ids, chunksize=2)):\n            q.put((sid, vol, mask))"
    new_loop = "for i, (sid, vol, mask, ok) in enumerate(pool.imap_unordered(_pre_one, test_ids, chunksize=2)):\n            if not ok:\n                pre_failed.add(sid)\n            q.put((sid, vol, mask))"
    if old_loop not in source:
        raise RuntimeError(f"{stem}: streaming-loop patch anchor missing")
    source = source.replace(old_loop, new_loop)
    marker = 'print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)'
    receipt = marker + f'''\n    _receipt = {{
        "reader": "{stem}", "rows": N, "preprocess_failures": len(pre_failed),
        "inference_failures": len(infer_failed), "failed_ids": sorted(pre_failed | infer_failed),
        "complete": done_n[0] == N and not pre_failed and not infer_failed,
    }}
    with open("/kaggle/working/{stem}_gold_receipt.json", "w") as _f:
        json.dump(_receipt, _f, indent=2)
    if not _receipt["complete"]:
        raise RuntimeError("{stem} gold inference had fallback studies: " + json.dumps(_receipt))'''
    if marker not in source:
        raise RuntimeError(f"{stem}: receipt patch anchor missing")
    return source.replace(marker, receipt)


def save_reader_cell(reader: str) -> str:
    return f'''from pathlib import Path
import json, shutil
WORK = Path("/kaggle/working")
shutil.copyfile(WORK / "ours_sub.csv", WORK / "{reader}_gold.csv")
shutil.copyfile(WORK / "submission_receipt.json", WORK / "{reader}_gold_receipt.json")
receipt = json.loads((WORK / "{reader}_gold_receipt.json").read_text())
if receipt.get("status") != "PASS" or int(receipt.get("fallback_studies", -1)) != 0:
    raise RuntimeError("{reader} did not pass without fallbacks: " + json.dumps(receipt))
if Path("/tmp/rsna_probe").exists():
    shutil.rmtree("/tmp/rsna_probe")
for candidate in (WORK / "ours_sub.csv", WORK / "submission_receipt.json"):
    candidate.unlink(missing_ok=True)
print("{reader.upper()}_GOLD_STAGE_SAVED", flush=True)
'''


PREP = r'''from pathlib import Path
import hashlib, json, os, shutil
import numpy as np
import pandas as pd

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
WORK = Path("/kaggle/working")
GOLD = WORK / "gold_eval"
if GOLD.exists():
    shutil.rmtree(GOLD)
GOLD.mkdir()

roots = []
for train_csv in Path("/kaggle/input").rglob("train.csv"):
    root = train_csv.parent
    if (root / "train_series.csv").exists() and (root / "train_series").is_dir():
        roots.append(root)
if len(roots) != 1:
    raise RuntimeError(f"expected one competition train root, found {roots}")
ROOT = roots[0]
train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
gold_mask = train[TARGETS].notna().all(axis=1)
gold = train.loc[gold_mask, ["StudyInstanceUID"] + TARGETS].copy()
if len(gold) != 58 or gold.StudyInstanceUID.nunique() != 58:
    raise RuntimeError("gold identity drift")
if not np.isin(gold[TARGETS].to_numpy(), [0, 1]).all():
    raise RuntimeError("gold labels are not binary")
series = pd.read_csv(ROOT / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
series = series[series.StudyInstanceUID.isin(set(gold.StudyInstanceUID))].copy()
if series.StudyInstanceUID.nunique() != 58:
    raise RuntimeError("gold series coverage incomplete")
gold[["StudyInstanceUID"]].to_csv(GOLD / "test.csv", index=False)
series.to_csv(GOLD / "test_series.csv", index=False)
sample = gold[["StudyInstanceUID"]].copy()
for target in TARGETS:
    sample[target] = 0.5
sample.to_csv(GOLD / "sample_submission.csv", index=False)
os.symlink(ROOT / "train_series", GOLD / "test_series", target_is_directory=True)
gold.to_csv(WORK / "gold_truth.csv", index=False)
print("GOLD_EVAL_READY", len(gold), len(series), ROOT, flush=True)
'''


COLLECT = r'''from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

WORK = Path("/kaggle/working")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
ARM_NAMES = ["r6_flip", "raptor384", "raptor224", "r7"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

truth_frame = pd.read_csv(WORK / "gold_truth.csv", dtype={"StudyInstanceUID": str})
ids = truth_frame.StudyInstanceUID.astype(str).tolist()
if len(ids) != 58 or len(set(ids)) != 58:
    raise RuntimeError("gold ID drift at collection")

files = {
    "r6_flip": WORK / "r6_flip_gold.csv",
    "raptor384": WORK / "raptor_raw.csv",
    "raptor224": WORK / "raptor224_raw.csv",
    "r7": WORK / "r7_gold.csv",
}
arms = []
for name in ARM_NAMES:
    frame = pd.read_csv(files[name], dtype={"StudyInstanceUID": str})
    if frame.columns.tolist() != ["StudyInstanceUID"] + TARGETS:
        raise RuntimeError(f"{name} schema mismatch")
    if sorted(frame.StudyInstanceUID) != sorted(ids) or frame.StudyInstanceUID.nunique() != 58:
        raise RuntimeError(f"{name} ID mismatch")
    values = frame.set_index("StudyInstanceUID").loc[ids, TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise RuntimeError(f"{name} invalid probabilities")
    arms.append(values)

receipts = {}
for name in ARM_NAMES:
    path = WORK / (name + "_gold_receipt.json")
    receipts[name] = json.loads(path.read_text())
for name in ("raptor384", "raptor224"):
    if not receipts[name].get("complete") or receipts[name].get("preprocess_failures") != 0 or receipts[name].get("inference_failures") != 0:
        raise RuntimeError(f"{name} fallback receipt failed")
for name in ("r6_flip", "r7"):
    if receipts[name].get("status") != "PASS" or receipts[name].get("fallback_studies") != 0:
        raise RuntimeError(f"{name} fallback receipt failed")

package_path = WORK / "multireader_gold_predictions.npz"
np.savez_compressed(
    package_path,
    ids=np.asarray(ids),
    targets=np.asarray(TARGETS),
    truth=truth_frame[TARGETS].to_numpy(np.int8),
    arm_names=np.asarray(ARM_NAMES),
    predictions=np.stack(arms, axis=1).astype(np.float32),
)
report = {
    "status": "PREDICTION_CACHE_COMPLETE",
    "study_count": 58,
    "unique_study_count": len(set(ids)),
    "target_count": 12,
    "arm_names": ARM_NAMES,
    "prediction_shape": [58, 4, 12],
    "finite": True,
    "probability_min": float(np.min(arms)),
    "probability_max": float(np.max(arms)),
    "receipts": receipts,
    "prediction_npz_sha256": sha256(package_path),
    "submission_created": False,
}
(WORK / "multireader_cache_report.json").write_text(json.dumps(report, indent=2) + "\n")

# No CSV is a retained output; the optimizer consumes only the immutable NPZ.
for path in WORK.rglob("*.csv"):
    path.unlink()
if list(WORK.rglob("submission.csv")):
    raise RuntimeError("forbidden submission artifact")
print(json.dumps(report, indent=2), flush=True)
'''


FINAL = r'''from pathlib import Path
import hashlib, json, subprocess, sys
WORK = Path("/kaggle/working")
optimizer = WORK / "optimize_nested_blend.py"
expected_optimizer_sha = "__OPTIMIZER_SHA__"
actual_optimizer_sha = hashlib.sha256(optimizer.read_bytes()).hexdigest()
if actual_optimizer_sha != expected_optimizer_sha:
    raise RuntimeError("optimizer identity mismatch")
rc = subprocess.run([
    sys.executable, str(optimizer), str(WORK / "multireader_gold_predictions.npz"),
    "--output", str(WORK / "multireader_nested_report.json")
]).returncode
if rc != 0:
    raise RuntimeError(f"optimizer failed with rc={rc}")
nested = json.loads((WORK / "multireader_nested_report.json").read_text())
cache = json.loads((WORK / "multireader_cache_report.json").read_text())
final = {
    "status": nested["status"],
    "cache_status": cache["status"],
    "source_notebook_sha256": {"r6": "__R6_SHA__", "r7": "__R7_SHA__"},
    "optimizer_sha256": actual_optimizer_sha,
    "prediction_npz_sha256": cache["prediction_npz_sha256"],
    "nested_delta": nested["nested_delta"],
    "gates": nested["gates"],
    "submission_created": False,
    "retained_csv_count": len(list(WORK.rglob("*.csv"))),
}
if final["retained_csv_count"] != 0 or list(WORK.rglob("submission.csv")):
    raise RuntimeError("CSV cleanup failed")
(WORK / "multireader_final_receipt.json").write_text(json.dumps(final, indent=2) + "\n")
print("MULTIREADER_NOTEBOOK_COMPLETE", json.dumps(final, indent=2), flush=True)
'''


def build(r6_path: Path, r7_path: Path, output: Path) -> None:
    if sha256(r6_path) != R6_SHA256:
        raise RuntimeError("R6 source notebook identity mismatch")
    if sha256(r7_path) != R7_SHA256:
        raise RuntimeError("R7 source notebook identity mismatch")
    r6 = json.loads(r6_path.read_text(encoding="utf-8"))
    r7 = json.loads(r7_path.read_text(encoding="utf-8"))
    optimizer_source = (HERE / "optimize_nested_blend.py").read_text(encoding="utf-8")
    optimizer_sha = hashlib.sha256(optimizer_source.encode()).hexdigest()

    r6_stage = patch_r6_frozen_mount(patch_gold_root(cell_source(r6, 6)))
    r7_stage = patch_gold_root(cell_source(r7, 6))
    final = (
        FINAL.replace("__OPTIMIZER_SHA__", optimizer_sha)
        .replace("__R6_SHA__", R6_SHA256)
        .replace("__R7_SHA__", R7_SHA256)
    )
    cells = [
        markdown("# RSNA knee four-reader nested blend screen\n\nPrivate, no-submission gold evaluation. Exact public-reader attribution is preserved from the two source notebooks. Raptor checkpoints are acknowledged as OAI-trained.\n"),
        code(PREP),
        code(patch_raptor_receipt(cell_source(r6, 2), "raptor384")),
        code(cell_source(r6, 3)),
        code(patch_raptor_receipt(cell_source(r6, 4), "raptor224")),
        code(cell_source(r6, 5)),
        code(r6_stage),
        code(cell_source(r6, 7)),
        code(save_reader_cell("r6_flip")),
        code(r7_stage),
        code(cell_source(r7, 7)),
        code(save_reader_cell("r7")),
        code(COLLECT),
        code("%%writefile /kaggle/working/optimize_nested_blend.py\n" + optimizer_source),
        code(final),
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
    parser.add_argument("--r6", type=Path, default=DEFAULT_R6)
    parser.add_argument("--r7", type=Path, default=DEFAULT_R7)
    parser.add_argument("--output", type=Path, default=HERE / "multireader-nested-blend.ipynb")
    args = parser.parse_args()
    build(args.r6, args.r7, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
