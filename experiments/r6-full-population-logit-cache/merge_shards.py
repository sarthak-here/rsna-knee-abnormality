"""Fail-closed merger for the two immutable R6 population shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--shard-0", type=Path, required=True)
    parser.add_argument("--shard-1", type=Path, required=True)
    parser.add_argument("--receipt-0", type=Path, required=True)
    parser.add_argument("--receipt-1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = pd.read_csv(args.train_csv, dtype={"StudyInstanceUID": str})[
        "StudyInstanceUID"
    ].astype(str).to_numpy()
    if expected.shape != (4407,) or len(set(expected.tolist())) != 4407:
        raise RuntimeError("official training identity drift")

    loaded = []
    for index, (cache_path, receipt_path) in enumerate(
        [(args.shard_0, args.receipt_0), (args.shard_1, args.receipt_1)]
    ):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "COMPLETE":
            raise RuntimeError(f"shard {index} receipt did not complete")
        if receipt.get("shard_index") != index or receipt.get("shard_count") != 2:
            raise RuntimeError(f"shard {index} receipt identity drift")
        if receipt.get("fallback_studies") != 0:
            raise RuntimeError(f"shard {index} contains fallback studies")
        if receipt.get("cache_sha256") != sha256(cache_path):
            raise RuntimeError(f"shard {index} cache hash drift")
        with np.load(cache_path, allow_pickle=False) as data:
            loaded.append({name: data[name].copy() for name in data.files})

    targets = loaded[0]["targets"]
    if not np.array_equal(targets, loaded[1]["targets"]) or targets.shape != (12,):
        raise RuntimeError("target identity drift")
    for index, data in enumerate(loaded):
        shard_expected = expected[index::2]
        if not np.array_equal(data["ids"].astype(str), shard_expected):
            raise RuntimeError(f"shard {index} ordering drift")
        count = shard_expected.size
        if data["probability"].shape != (count, 12):
            raise RuntimeError(f"shard {index} probability shape drift")
        if data["logits"].shape != (count, 4, 2, 12):
            raise RuntimeError(f"shard {index} logit shape drift")
        if not np.isfinite(data["probability"]).all() or not np.isfinite(data["logits"]).all():
            raise RuntimeError(f"shard {index} contains non-finite values")
        if data["probability"].min() < 0 or data["probability"].max() > 1:
            raise RuntimeError(f"shard {index} probability range drift")

    probability = np.empty((4407, 12), np.float32)
    logits = np.empty((4407, 4, 2, 12), np.float32)
    for index, data in enumerate(loaded):
        probability[index::2] = data["probability"]
        logits[index::2] = data["logits"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=expected.astype("U80"),
        targets=targets.astype("U32"),
        probability=probability,
        logits=logits,
    )
    result = {
        "status": "COMPLETE",
        "studies": 4407,
        "targets": 12,
        "probability_shape": [4407, 12],
        "logit_shape": [4407, 4, 2, 12],
        "finite": True,
        "fallback_studies": 0,
        "shard_sha256": [sha256(args.shard_0), sha256(args.shard_1)],
        "cache_sha256": sha256(args.output),
    }
    receipt_path = args.output.with_name("r6_full_population_receipt.json")
    receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
