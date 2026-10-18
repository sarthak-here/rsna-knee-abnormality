#!/usr/bin/env python3
"""Evaluate the predeclared R6 stride-2 replacement on the frozen anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


EXPECTED_ANCHOR_SHA256 = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"
EXPECTED_ARMS = ["r6_flip", "raptor384", "raptor224", "r7"]
WEIGHTS = np.asarray([0.50, 0.325, 0.175, 0.0], np.float64)


def rank_columns(x: np.ndarray) -> np.ndarray:
    return np.stack(
        [rankdata(x[:, j], method="average") / len(x) for j in range(x.shape[1])],
        axis=1,
    )


def blend(predictions: np.ndarray) -> np.ndarray:
    ranked = np.stack(
        [rank_columns(predictions[:, arm]) for arm in range(predictions.shape[1])],
        axis=1,
    )
    return np.einsum("nkt,k->nt", ranked, WEIGHTS)


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    values = np.asarray([roc_auc_score(y[:, j], p[:, j]) for j in range(12)])
    return float(values.mean()), values


def bootstrap(y: np.ndarray, a: np.ndarray, c: np.ndarray) -> dict:
    def sampled_auc(yy: np.ndarray, pp: np.ndarray) -> float | None:
        scores = []
        for j in range(12):
            target = yy[:, j]
            pos = int(target.sum())
            neg = len(target) - pos
            if pos == 0 or neg == 0:
                return None
            ranks = rankdata(pp[:, j], method="average")
            scores.append(
                (ranks[target == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
            )
        return float(np.mean(scores))

    rng = np.random.default_rng(20261018)
    values = []
    for _ in range(10000):
        idx = rng.integers(0, len(y), len(y))
        av, cv = sampled_auc(y[idx], a[idx]), sampled_auc(y[idx], c[idx])
        if av is not None and cv is not None:
            values.append(cv - av)
    d = np.asarray(values)
    if not len(d):
        raise ValueError("no valid bootstrap samples")
    return {
        "valid_samples": int(len(d)),
        "probability_positive": float((d > 0).mean()),
        "q025": float(np.quantile(d, 0.025)),
        "q50": float(np.quantile(d, 0.50)),
        "q975": float(np.quantile(d, 0.975)),
    }


def evaluate(anchor_path: Path, dense_path: Path) -> dict:
    anchor_sha = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
    if anchor_sha != EXPECTED_ANCHOR_SHA256:
        raise ValueError(f"anchor SHA-256 drift: {anchor_sha}")
    with np.load(anchor_path, allow_pickle=False) as z:
        ids = z["ids"]
        arms = z["arm_names"].tolist()
        truth = z["truth"].astype(np.int8)
        predictions = z["predictions"].astype(np.float64)
    with np.load(dense_path, allow_pickle=False) as z:
        dense_ids = z["ids"]
        dense = z["predictions"].astype(np.float64)
    if ids.dtype.kind not in "US" or len(ids) != 58 or len(np.unique(ids)) != 58:
        raise ValueError("anchor IDs are not 58 unique Unicode values")
    if arms != EXPECTED_ARMS or predictions.shape != (58, 4, 12):
        raise ValueError("anchor arm contract drift")
    if dense_ids.dtype.kind not in "US" or dense_ids.tolist() != ids.tolist():
        raise ValueError("dense-window ID/order mismatch")
    if dense.shape != (58, 12) or not np.isfinite(dense).all():
        raise ValueError("dense-window predictions invalid")
    if np.any((dense < 0) | (dense > 1)):
        raise ValueError("dense-window probabilities outside [0,1]")

    candidate_arms = predictions.copy()
    candidate_arms[:, 0] = dense
    anchor = blend(predictions)
    candidate = blend(candidate_arms)
    am, at = macro_auc(truth, anchor)
    cm, ct = macro_auc(truth, candidate)
    delta = ct - at
    boot = bootstrap(truth, anchor, candidate)
    gates = {
        "macro_gain_ge_0.001": bool(cm - am >= 0.001),
        "nonnegative_targets_ge_8": bool((delta >= -1e-12).sum() >= 8),
        "bootstrap_p_ge_0.90": bool(boot["probability_positive"] >= 0.90),
        "bootstrap_q025_gt_minus_0.001": bool(boot["q025"] > -0.001),
        "finite_predictions": True,
        "identity_integrity": True,
    }
    return {
        "status": "PASS" if all(gates.values()) else "REJECTED",
        "contract": "r6_eval_stride_2_replacement_v1",
        "anchor_sha256": anchor_sha,
        "dense_prediction_sha256": hashlib.sha256(dense_path.read_bytes()).hexdigest(),
        "anchor_macro_auc": am,
        "candidate_macro_auc": cm,
        "macro_delta": cm - am,
        "target_delta": delta.tolist(),
        "nonnegative_targets": int((delta >= -1e-12).sum()),
        "bootstrap": boot,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("anchor", type=Path)
    parser.add_argument("dense", type=Path)
    parser.add_argument("--out", type=Path, default=Path("r6_dense_window_report.json"))
    args = parser.parse_args()
    report = evaluate(args.anchor, args.dense)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
