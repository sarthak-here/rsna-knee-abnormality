"""Build the exact no-submission R6 full-population fold-logit cache."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MULTI_PATH = HERE.parent / "multireader-nested-blend" / "build_notebook.py"
ROUTER_PATH = HERE.parent / "uncertainty-router" / "build_notebook.py"

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

BASE = load_module("multireader_builder", MULTI_PATH)
ROUTER = load_module("uncertainty_router_builder", ROUTER_PATH)

PREP = r'''from pathlib import Path
import os, shutil, time
import pandas as pd

NB_T0 = time.time()
WORK = Path("/kaggle/working")
EVAL = WORK / "all_train_eval"
if EVAL.exists():
    shutil.rmtree(EVAL)
EVAL.mkdir()

roots = []
for train_csv in Path("/kaggle/input").rglob("train.csv"):
    root = train_csv.parent
    if (root / "train_series.csv").exists() and (root / "train_series").is_dir():
        roots.append(root)
if len(roots) != 1:
    raise RuntimeError(f"expected one competition train root, found {roots}")
root = roots[0]
train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
all_ids = train[["StudyInstanceUID"]].copy()
if len(all_ids) != 4407 or all_ids.StudyInstanceUID.nunique() != 4407:
    raise RuntimeError("full-population study identity drift")
SHARD_INDEX = __SHARD_INDEX__
SHARD_COUNT = __SHARD_COUNT__
ids = all_ids.iloc[SHARD_INDEX::SHARD_COUNT].reset_index(drop=True)
EXPECTED = __EXPECTED__
if len(ids) != EXPECTED or ids.StudyInstanceUID.nunique() != EXPECTED:
    raise RuntimeError("deterministic shard identity drift")
series = pd.read_csv(root / "train_series.csv", dtype=str)
series = series[series.StudyInstanceUID.isin(set(ids.StudyInstanceUID))].copy()
if set(series.StudyInstanceUID) != set(ids.StudyInstanceUID):
    raise RuntimeError("shard series coverage drift")
ids.to_csv(EVAL / "test.csv", index=False)
series.to_csv(EVAL / "test_series.csv", index=False)
sample = ids.copy()
for target in ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
               "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
               "Contusion", "Fracture"]:
    sample[target] = 0.5
sample.to_csv(EVAL / "sample_submission.csv", index=False)
os.symlink(root / "train_series", EVAL / "test_series", target_is_directory=True)
print("FULL_POPULATION_SHARD_READY", SHARD_INDEX, SHARD_COUNT, len(ids), len(series), root, flush=True)
'''

COLLECT = r'''from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

WORK = Path("/kaggle/working")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
PERM = np.asarray([0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

frame = pd.read_csv(WORK / "ours_sub.csv", dtype={"StudyInstanceUID": str})
if frame.columns.tolist() != ["StudyInstanceUID"] + TARGETS:
    raise RuntimeError("R6 probability schema drift")
ids = frame.StudyInstanceUID.astype(str).tolist()
EXPECTED = __EXPECTED__
SHARD_INDEX = __SHARD_INDEX__
SHARD_COUNT = __SHARD_COUNT__
if len(ids) != EXPECTED or len(set(ids)) != EXPECTED:
    raise RuntimeError("R6 shard ID drift")
probability = frame[TARGETS].to_numpy(np.float32)
if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
    raise RuntimeError("R6 probability drift")

parts = sorted(WORK.glob("r6_fold_logits_worker*.json"))
if len(parts) != 2:
    raise RuntimeError(f"expected two worker logit dumps, found {parts}")
raw = {}
for path in parts:
    part = json.loads(path.read_text())
    overlap = set(raw).intersection(part)
    if overlap:
        raise RuntimeError(f"duplicate worker IDs: {sorted(overlap)[:3]}")
    raw.update(part)
if set(raw) != set(ids):
    raise RuntimeError("R6 logit identity drift")
logits = np.empty((EXPECTED, 4, 2, 12), np.float32)
for index, uid in enumerate(ids):
    record = raw[uid]
    plain = np.asarray(record.get("z_plain"), np.float32)
    mirror_raw = np.asarray(record.get("z_mirror_raw"), np.float32)
    if plain.shape != (4, 12) or mirror_raw.shape != (4, 12):
        raise RuntimeError(f"fold/pass shape drift for {uid}")
    mirror = mirror_raw[:, PERM]
    mirror[:, 1] = plain[:, 1]
    logits[index, :, 0] = plain
    logits[index, :, 1] = mirror
if not np.isfinite(logits).all():
    raise RuntimeError("non-finite R6 logits")
receipt = json.loads((WORK / "submission_receipt.json").read_text())
if receipt.get("status") != "PASS" or int(receipt.get("fallback_studies", -1)) != 0:
    raise RuntimeError("R6 inference receipt failed")

cache = WORK / f"r6_full_population_logits_shard{SHARD_INDEX}.npz"
np.savez_compressed(cache, ids=np.asarray(ids, dtype="U80"),
                    targets=np.asarray(TARGETS, dtype="U32"),
                    probability=probability, logits=logits)
result = {
    "status": "COMPLETE", "shard_index": SHARD_INDEX, "shard_count": SHARD_COUNT,
    "studies": EXPECTED, "targets": 12,
    "probability_shape": [EXPECTED, 12], "logit_shape": [EXPECTED, 4, 2, 12],
    "finite": True, "fallback_studies": 0, "receipt": receipt,
    "cache_sha256": sha256(cache), "submission_created": False,
}
(WORK / f"r6_full_population_receipt_shard{SHARD_INDEX}.json").write_text(json.dumps(result, indent=2) + "\n")
for path in WORK.rglob("*.csv"):
    path.unlink()
if list(WORK.rglob("*.csv")):
    raise RuntimeError("forbidden CSV artifact")
print("R6_FULL_POPULATION_COMPLETE", json.dumps(result, indent=2), flush=True)
'''

def patch_all_root(source: str) -> str:
    source = source.replace("ROOT = find_test_root()", 'ROOT = "/kaggle/working/all_train_eval"')
    source = source.replace("DATA = find_mount(COMPETITION, 'test.csv')",
                            "DATA = Path('/kaggle/working/all_train_eval')")
    return source

def render(template: str, shard_index: int, shard_count: int) -> str:
    expected = len(range(shard_index, 4407, shard_count))
    return (template.replace("__SHARD_INDEX__", str(shard_index))
                    .replace("__SHARD_COUNT__", str(shard_count))
                    .replace("__EXPECTED__", str(expected)))


def build(r6_path: Path, output: Path, shard_index: int, shard_count: int) -> None:
    if BASE.sha256(r6_path) != BASE.R6_SHA256:
        raise RuntimeError("R6 source notebook identity mismatch")
    r6 = json.loads(r6_path.read_text(encoding="utf-8"))
    stage = BASE.patch_r6_frozen_mount(patch_all_root(BASE.cell_source(r6, 6)))
    if stage.count("DEADLINE_HOURS = 8.3") != 1:
        raise RuntimeError("R6 deadline anchor missing or ambiguous")
    # The original blend reserved time for two Raptor stages. This isolated R6
    # cache uses the Kaggle runtime directly and needs the full-population bound.
    stage = stage.replace("DEADLINE_HOURS = 8.3", "DEADLINE_HOURS = 11.0", 1)
    stage = ROUTER.patch_r6_logit_dump(stage)
    cells = [
        BASE.markdown("# R6 full-population fold-logit cache\n\nPrivate inference-only cache; no labels and no submission.\n"),
        BASE.code(render(PREP, shard_index, shard_count)), BASE.code(stage),
        BASE.code(BASE.cell_source(r6, 7)), BASE.code(render(COLLECT, shard_index, shard_count)),
    ]
    notebook = {"cells": cells, "metadata": r6.get("metadata", {}),
                "nbformat": 4, "nbformat_minor": 5}
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6", type=Path, default=BASE.DEFAULT_R6)
    parser.add_argument("--shard-count", type=int, default=2)
    args = parser.parse_args()
    if args.shard_count != 2:
        raise RuntimeError("the repaired runtime contract is pinned to exactly two shards")
    for shard_index in range(args.shard_count):
        shard_dir = HERE / f"shard-{shard_index}"
        shard_dir.mkdir(exist_ok=True)
        output = shard_dir / "r6-full-population-logit-cache.ipynb"
        build(args.r6, output, shard_index, args.shard_count)
        metadata = {
            "id": f"sarthaksharma14/rsna-knee-r6-full-logits-shard-{shard_index}",
            "title": f"RSNA Knee R6 Full Logits Shard {shard_index}",
            "code_file": output.name, "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1",
                                "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "kernel_sources": [], "model_sources": [],
        }
        (shard_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(output)

if __name__ == "__main__":
    main()
