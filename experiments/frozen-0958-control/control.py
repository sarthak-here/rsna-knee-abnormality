"""Exact zero-failure 57483292 blend; no fitting, labels, or fallback."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture")
RAPTOR_WEIGHTS = np.array([.5, .5, .5, .5, .5, .2, .6, .5, .8, .35, .5, .5])


def blend(r6, raptor384, raptor224):
    arrays = [np.asarray(x, dtype=np.float64) for x in (r6, raptor384, raptor224)]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] != 12:
        raise ValueError("expected nonempty [study,12] arrays")
    for x in arrays:
        if x.shape != shape or not np.isfinite(x).all() or not ((0 <= x) & (x <= 1)).all():
            raise ValueError("reader shape/range/finiteness mismatch")
    if any((x == .5).all(axis=1).any() for x in arrays[1:]):
        raise ValueError("source Raptor fallback sentinel; zero-failure control required")
    ranks = [pd.DataFrame(x).rank(method="average", pct=True).to_numpy() for x in arrays]
    raptor_half = .65 * ranks[1] + .35 * ranks[2]
    return (RAPTOR_WEIGHTS * raptor_half + (1 - RAPTOR_WEIGHTS) * ranks[0]).astype(np.float32)


def read_preview(path, ids=None):
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if tuple(frame.columns) != ("StudyInstanceUID",) + TARGETS:
        raise ValueError("target columns/order mismatch")
    found = frame.StudyInstanceUID.tolist()
    if len(set(found)) != len(found) or not found:
        raise ValueError("duplicate/empty study IDs")
    if ids is not None and found != ids:
        raise ValueError("reader study order mismatch")
    return found, frame.iloc[:, 1:].to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preview_dir", type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    paths = [args.preview_dir / n for n in ("ours_sub.csv", "raptor_raw.csv", "raptor224_raw.csv", "submission.csv")]
    ids, r6 = read_preview(paths[0])
    arrays = [read_preview(p, ids)[1] for p in paths[1:]]
    actual = blend(r6, arrays[0], arrays[1])
    expected = arrays[2].astype(np.float32)
    if not np.array_equal(actual, expected):
        raise ValueError(f"scored preview mismatch: {np.abs(actual - expected).max()}")
    receipt = {
        "control_submission": 57483292, "public_score": "0.958",
        "rows": len(ids), "target_order": list(TARGETS),
        "preview_bitwise_float32_equal": True, "max_error": 0.0,
        "formula": "w*(0.65*avg_tie_pct_rank(Raptor384)+0.35*avg_tie_pct_rank(Raptor224))+(1-w)*avg_tie_pct_rank(R6)",
        "raptor_per_target_weights": RAPTOR_WEIGHTS.tolist(),
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "labels_read": False, "auc_computed": False, "model_fitted": False,
        "new_predictions_generated": False, "gpu_used": False,
        "scope": "Existing public three-study preview arithmetic only, not hidden or pathology validation"
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
