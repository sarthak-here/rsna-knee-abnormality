#!/usr/bin/env python3
"""Evaluate the predeclared label-free uncertainty router."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


EXPECTED_TARGETS = 12
RAPTOR_224_SHARE = 0.35
ROUTER_SCALE = 0.20
ROUTER_LO = 0.45
ROUTER_HI = 0.55


def rank_columns(x: np.ndarray) -> np.ndarray:
    return np.stack(
        [rankdata(x[:, j], method="average") / len(x) for j in range(x.shape[1])], axis=1
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    scores = np.array([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])])
    return float(scores.mean()), scores


def build_predictions(
    r6_logits: np.ndarray, raptor_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if r6_logits.ndim != 4 or r6_logits.shape[1:] != (4, 2, EXPECTED_TARGETS):
        raise ValueError(f"r6_logits must have shape [N,4,2,12], got {r6_logits.shape}")
    if raptor_prob.shape != (len(r6_logits), 2, EXPECTED_TARGETS):
        raise ValueError(f"raptor_prob must have shape [N,2,12], got {raptor_prob.shape}")
    if not np.isfinite(r6_logits).all() or not np.isfinite(raptor_prob).all():
        raise ValueError("non-finite model output")
    if np.any((raptor_prob < 0) | (raptor_prob > 1)):
        raise ValueError("Raptor probabilities outside [0,1]")

    fold_logits = r6_logits.mean(axis=2)
    r6_prob = sigmoid(fold_logits.mean(axis=1))
    r6_unc = fold_logits.std(axis=1)

    r384_rank = rank_columns(raptor_prob[:, 0])
    r224_rank = rank_columns(raptor_prob[:, 1])
    raptor_rank = rank_columns(
        (1.0 - RAPTOR_224_SHARE) * r384_rank + RAPTOR_224_SHARE * r224_rank
    )
    raptor_unc = np.abs(logit(raptor_prob[:, 0]) - logit(raptor_prob[:, 1]))

    r6_rank = rank_columns(r6_prob)
    r6_unc_rank = rank_columns(r6_unc)
    raptor_unc_rank = rank_columns(raptor_unc)
    raptor_share = np.clip(
        0.50 + ROUTER_SCALE * (r6_unc_rank - raptor_unc_rank), ROUTER_LO, ROUTER_HI
    )
    anchor = rank_columns(0.50 * r6_rank + 0.50 * raptor_rank)
    candidate = rank_columns(
        (1.0 - raptor_share) * r6_rank + raptor_share * raptor_rank
    )
    return anchor, candidate, raptor_share


def bootstrap(
    y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray, n_boot: int = 10000
) -> dict:
    rng = np.random.default_rng(20261018)
    delta = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            a, _ = macro_auc(y[idx], anchor[idx])
            c, _ = macro_auc(y[idx], candidate[idx])
        except ValueError:
            continue
        delta.append(c - a)
    if not delta:
        raise ValueError("no valid paired bootstrap samples")
    d = np.asarray(delta)
    return {
        "valid_samples": int(len(d)),
        "probability_positive": float((d > 0).mean()),
        "q025": float(np.quantile(d, 0.025)),
        "q50": float(np.quantile(d, 0.5)),
        "q975": float(np.quantile(d, 0.975)),
    }


def evaluate(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as z:
        required = {"ids", "truth", "r6_logits", "raptor_prob"}
        if not required.issubset(z.files):
            raise ValueError(f"missing keys: {sorted(required - set(z.files))}")
        ids = z["ids"]
        truth = z["truth"].astype(np.int8)
        r6_logits = z["r6_logits"].astype(np.float64)
        raptor_prob = z["raptor_prob"].astype(np.float64)
    if ids.dtype.kind not in "US" or len(ids) != 58 or len(np.unique(ids)) != 58:
        raise ValueError("expected 58 unique Unicode IDs")
    if truth.shape != (58, EXPECTED_TARGETS) or not np.isin(truth, [0, 1]).all():
        raise ValueError("truth must be binary [58,12]")

    anchor, candidate, shares = build_predictions(r6_logits, raptor_prob)
    anchor_macro, anchor_targets = macro_auc(truth, anchor)
    candidate_macro, candidate_targets = macro_auc(truth, candidate)
    target_delta = candidate_targets - anchor_targets
    boot = bootstrap(truth, anchor, candidate)
    gates = {
        "macro_gain_ge_0.001": bool(candidate_macro - anchor_macro >= 0.001),
        "nonnegative_targets_ge_8": bool((target_delta >= 0).sum() >= 8),
        "bootstrap_p_ge_0.90": bool(boot["probability_positive"] >= 0.90),
        "bootstrap_q025_gt_minus_0.001": bool(boot["q025"] > -0.001),
        "router_nonconstant": bool(float(shares.max() - shares.min()) > 0),
        "router_bounds": bool(shares.min() >= ROUTER_LO and shares.max() <= ROUTER_HI),
    }
    return {
        "contract": "fold_uncertainty_router_v1",
        "input_sha256": digest,
        "counts": {"studies": 58, "targets": EXPECTED_TARGETS},
        "anchor_macro_auc": anchor_macro,
        "candidate_macro_auc": candidate_macro,
        "macro_delta": candidate_macro - anchor_macro,
        "target_delta": target_delta.tolist(),
        "nonnegative_targets": int((target_delta >= 0).sum()),
        "raptor_share": {
            "min": float(shares.min()),
            "max": float(shares.max()),
            "mean": float(shares.mean()),
        },
        "bootstrap": boot,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_cache", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("uncertainty_router_report.json")
    )
    args = parser.parse_args()
    report = evaluate(args.prediction_cache)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
