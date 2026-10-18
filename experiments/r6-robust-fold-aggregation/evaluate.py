#!/usr/bin/env python3
"""Evaluate the predeclared four-fold 25% trimmed-mean R6 aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


EXPECTED_CACHE_SHA256 = "92ef1f26200069ae7b3e7e679b59bf30ed3556cf669ccdfe40fccb0cb1046623"
STACKER_TARGETS = (7, 9)  # Effusion and Baker's: preserve published postprocessors.
RAPTOR_224_SHARE = 0.35


def rank_columns(x: np.ndarray) -> np.ndarray:
    return np.stack(
        [rankdata(x[:, j], method="average") / len(x) for j in range(x.shape[1])],
        axis=1,
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    per_target = np.asarray(
        [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]
    )
    return float(per_target.mean()), per_target


def predictions(
    r6_probability: np.ndarray, r6_logits: np.ndarray, raptor_probability: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    if r6_probability.shape != (58, 12):
        raise ValueError(f"r6_probability shape drift: {r6_probability.shape}")
    if r6_logits.shape != (58, 4, 2, 12):
        raise ValueError(f"r6_logits shape drift: {r6_logits.shape}")
    if raptor_probability.shape != (58, 2, 12):
        raise ValueError(f"raptor_probability shape drift: {raptor_probability.shape}")
    if not all(
        np.isfinite(x).all()
        for x in (r6_probability, r6_logits, raptor_probability)
    ):
        raise ValueError("non-finite input")

    # This must reproduce the published non-stacker outputs before testing an
    # alternative fold reducer; otherwise the logit cache contract has drifted.
    published_from_logits = sigmoid(r6_logits.mean(axis=(1, 2)))
    ordinary_targets = [j for j in range(12) if j not in STACKER_TARGETS]
    reconstruction_max_abs = float(
        np.max(
            np.abs(
                published_from_logits[:, ordinary_targets]
                - r6_probability[:, ordinary_targets]
            )
        )
    )

    per_fold = r6_logits.mean(axis=2)
    ordered = np.sort(per_fold, axis=1)
    robust_r6 = sigmoid(ordered[:, 1:3].mean(axis=1))
    robust_r6[:, STACKER_TARGETS] = r6_probability[:, STACKER_TARGETS]

    r384 = rank_columns(raptor_probability[:, 0])
    r224 = rank_columns(raptor_probability[:, 1])
    raptor = (1.0 - RAPTOR_224_SHARE) * r384 + RAPTOR_224_SHARE * r224
    anchor = 0.50 * rank_columns(r6_probability) + 0.50 * raptor
    candidate = 0.50 * rank_columns(robust_r6) + 0.50 * raptor
    return anchor, candidate, reconstruction_max_abs


def bootstrap(y: np.ndarray, a: np.ndarray, c: np.ndarray) -> dict:
    def sampled_auc(yy: np.ndarray, pp: np.ndarray) -> float | None:
        scores = []
        for j in range(yy.shape[1]):
            target = yy[:, j]
            positives = int(target.sum())
            negatives = len(target) - positives
            if positives == 0 or negatives == 0:
                return None
            ranks = rankdata(pp[:, j], method="average")
            scores.append(
                (ranks[target == 1].sum() - positives * (positives + 1) / 2)
                / (positives * negatives)
            )
        return float(np.mean(scores))

    rng = np.random.default_rng(20261018)
    values = []
    for _ in range(10000):
        idx = rng.integers(0, len(y), len(y))
        av = sampled_auc(y[idx], a[idx])
        cv = sampled_auc(y[idx], c[idx])
        if av is None or cv is None:
            continue
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


def evaluate(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_CACHE_SHA256:
        raise ValueError(f"cache SHA-256 drift: {digest}")
    with np.load(path, allow_pickle=False) as z:
        ids = z["ids"]
        y = z["truth"].astype(np.int8)
        r6 = z["r6_prob"].astype(np.float64)
        logits = z["r6_logits"].astype(np.float64)
        raptor = z["raptor_prob"].astype(np.float64)
    if ids.dtype.kind not in "US" or len(ids) != 58 or len(np.unique(ids)) != 58:
        raise ValueError("expected 58 unique Unicode IDs")
    if y.shape != (58, 12) or not np.isin(y, (0, 1)).all():
        raise ValueError("truth must be binary [58,12]")

    anchor, candidate, reconstruction = predictions(r6, logits, raptor)
    anchor_macro, anchor_target = macro_auc(y, anchor)
    candidate_macro, candidate_target = macro_auc(y, candidate)
    delta = candidate_target - anchor_target
    boot = bootstrap(y, anchor, candidate)
    gates = {
        "macro_gain_ge_0.001": bool(candidate_macro - anchor_macro >= 0.001),
        "nonnegative_targets_ge_8": bool((delta >= -1e-12).sum() >= 8),
        "bootstrap_p_ge_0.90": bool(boot["probability_positive"] >= 0.90),
        "bootstrap_q025_gt_minus_0.001": bool(boot["q025"] > -0.001),
        "mean_logit_reconstruction_max_abs_le_1e-6": bool(reconstruction <= 1e-6),
        "finite_outputs": bool(np.isfinite(anchor).all() and np.isfinite(candidate).all()),
    }
    return {
        "status": "PASS" if all(gates.values()) else "REJECTED",
        "contract": "r6_four_fold_middle_two_logit_mean_v1",
        "input_sha256": digest,
        "stacker_targets_preserved": list(STACKER_TARGETS),
        "anchor_macro_auc": anchor_macro,
        "candidate_macro_auc": candidate_macro,
        "macro_delta": candidate_macro - anchor_macro,
        "target_delta": delta.tolist(),
        "nonnegative_targets": int((delta >= -1e-12).sum()),
        "mean_logit_reconstruction_max_abs": reconstruction,
        "bootstrap": boot,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--out", type=Path, default=Path("r6_robust_fold_report.json"))
    args = parser.parse_args()
    report = evaluate(args.cache)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
