"""Nested, shrinkage-constrained per-finding blend screen.

The input NPZ is produced by a separate no-submission inference notebook and
must contain exactly these arrays:

* ids: shape [58], unique study identifiers
* targets: shape [12], target names
* truth: shape [58, 12], binary organizer labels
* arm_names: shape [4], exact reader names
* predictions: shape [58, 4, 12], finite probabilities

This script never creates a competition submission.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXPECTED_ARMS = ("r6_flip", "raptor384", "raptor224", "r7")
ANCHOR = np.asarray([0.50, 0.325, 0.175, 0.0], dtype=np.float64)
GRID_STEP = 0.10
L2_PENALTY = 0.01
OUTER_SPLITS = 5
BOOTSTRAP_DRAWS = 5000
SEED = 20261017


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    """Stable average-free ranks scaled to [0, 1]."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def simplex_grid(step: float = GRID_STEP) -> np.ndarray:
    units = int(round(1.0 / step))
    rows = []
    for cuts in itertools.combinations_with_replacement(range(4), units):
        counts = np.bincount(cuts, minlength=4)
        rows.append(counts / units)
    grid = np.unique(np.asarray(rows, dtype=np.float64), axis=0)
    if not np.allclose(grid.sum(axis=1), 1.0):
        raise RuntimeError("simplex grid construction failed")
    return grid


def macro_auc(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for target in range(truth.shape[1]):
        if np.unique(truth[:, target]).size == 2:
            scores.append(roc_auc_score(truth[:, target], prediction[:, target]))
    if len(scores) != truth.shape[1]:
        raise RuntimeError("macro AUC lost a target")
    return float(np.mean(scores))


def nearest_grid(weight: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return grid[np.argmin(((grid - weight[None, :]) ** 2).sum(axis=1))]


def choose_weight(
    truth: np.ndarray,
    ranked_arms: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    scores = np.empty(len(grid), dtype=np.float64)
    for index, weight in enumerate(grid):
        blend = ranked_arms @ weight
        auc = roc_auc_score(truth, blend)
        scores[index] = auc - L2_PENALTY * float(((weight - ANCHOR) ** 2).sum())
    best = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12))
    if len(best) > 1:
        distance = ((grid[best] - ANCHOR[None, :]) ** 2).sum(axis=1)
        return grid[best[np.argmin(distance)]]
    return grid[best[0]]


def paired_bootstrap(
    truth: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 1)
    deltas = []
    while len(deltas) < BOOTSTRAP_DRAWS:
        sample = rng.integers(0, len(truth), len(truth))
        try:
            delta = macro_auc(truth[sample], candidate[sample]) - macro_auc(
                truth[sample], anchor[sample]
            )
        except (RuntimeError, ValueError):
            continue
        deltas.append(delta)
    values = np.asarray(deltas)
    return {
        "draws": BOOTSTRAP_DRAWS,
        "mean_delta": float(values.mean()),
        "p_improve": float((values > 0).mean()),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def validate(package: np.lib.npyio.NpzFile) -> tuple[np.ndarray, ...]:
    expected = {"ids", "targets", "truth", "arm_names", "predictions"}
    if set(package.files) != expected:
        raise RuntimeError(f"NPZ schema mismatch: {package.files}")
    ids = package["ids"].astype(str)
    targets = package["targets"].astype(str)
    truth = package["truth"].astype(np.float64)
    arm_names = package["arm_names"].astype(str)
    predictions = package["predictions"].astype(np.float64)
    if ids.shape != (58,) or len(set(ids)) != 58:
        raise RuntimeError("expected 58 unique gold IDs")
    if targets.shape != (12,) or len(set(targets)) != 12:
        raise RuntimeError("expected 12 unique targets")
    if tuple(arm_names) != EXPECTED_ARMS:
        raise RuntimeError(f"arm identity mismatch: {arm_names.tolist()}")
    if truth.shape != (58, 12) or not np.isin(truth, [0, 1]).all():
        raise RuntimeError("truth must be binary [58, 12]")
    if predictions.shape != (58, 4, 12):
        raise RuntimeError("predictions must have shape [58, 4, 12]")
    if not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any():
        raise RuntimeError("invalid reader probabilities")
    for target in range(12):
        positives = int(truth[:, target].sum())
        if min(positives, 58 - positives) < OUTER_SPLITS:
            raise RuntimeError(f"target {targets[target]} cannot support five folds")
    return ids, targets, truth, arm_names, predictions


def run(input_path: Path, output_path: Path) -> dict:
    with np.load(input_path, allow_pickle=False) as package:
        ids, targets, truth, arm_names, predictions = validate(package)

    ranked = np.empty_like(predictions)
    for arm in range(4):
        for target in range(12):
            ranked[:, arm, target] = rank01(predictions[:, arm, target])

    grid = simplex_grid()
    anchor_prediction = np.einsum("nat,a->nt", ranked, ANCHOR)
    candidate_prediction = np.full((58, 12), np.nan, dtype=np.float64)
    fold_index = np.full((58, 12), -1, dtype=np.int16)
    fold_weights = np.empty((12, OUTER_SPLITS, 4), dtype=np.float64)
    target_rows = []

    for target in range(12):
        splitter = StratifiedKFold(
            n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED + target
        )
        for fold, (fit, heldout) in enumerate(splitter.split(ids, truth[:, target])):
            weight = choose_weight(truth[fit, target], ranked[fit, :, target], grid)
            fold_weights[target, fold] = weight
            candidate_prediction[heldout, target] = ranked[heldout, :, target] @ weight
            fold_index[heldout, target] = fold
        if np.isnan(candidate_prediction[:, target]).any() or (fold_index[:, target] < 0).any():
            raise RuntimeError(f"incomplete outer coverage for {targets[target]}")
        anchor_auc = roc_auc_score(truth[:, target], anchor_prediction[:, target])
        candidate_auc = roc_auc_score(truth[:, target], candidate_prediction[:, target])
        target_rows.append(
            {
                "target": targets[target],
                "anchor_auc": float(anchor_auc),
                "nested_candidate_auc": float(candidate_auc),
                "delta": float(candidate_auc - anchor_auc),
                "outer_weights": fold_weights[target].tolist(),
                "r7_positive_outer_folds": int((fold_weights[target, :, 3] > 0).sum()),
            }
        )

    final_weights = np.stack(
        [nearest_grid(np.median(fold_weights[target], axis=0), grid) for target in range(12)]
    )
    anchor_auc = macro_auc(truth, anchor_prediction)
    candidate_auc = macro_auc(truth, candidate_prediction)
    delta = candidate_auc - anchor_auc
    bootstrap = paired_bootstrap(truth, anchor_prediction, candidate_prediction)

    outer_deltas = []
    for fold in range(OUTER_SPLITS):
        fold_candidate, fold_anchor, fold_truth = [], [], []
        for target in range(12):
            heldout = fold_index[:, target] == fold
            fold_candidate.append(candidate_prediction[heldout, target])
            fold_anchor.append(anchor_prediction[heldout, target])
            fold_truth.append(truth[heldout, target])
        per_target_delta = [
            roc_auc_score(y, p1) - roc_auc_score(y, p0)
            for y, p0, p1 in zip(fold_truth, fold_anchor, fold_candidate)
        ]
        outer_deltas.append(float(np.mean(per_target_delta)))

    gates = {
        "nested_delta_at_least_0_002": delta >= 0.002,
        "nonnegative_targets_at_least_8": sum(row["delta"] >= 0 for row in target_rows) >= 8,
        "positive_outer_folds_at_least_4": sum(value > 0 for value in outer_deltas) >= 4,
        "bootstrap_p_at_least_0_90": bootstrap["p_improve"] >= 0.90,
        "bootstrap_q025_above_minus_0_001": bootstrap["q025"] > -0.001,
        "r7_supported_by_three_outer_folds": any(
            row["r7_positive_outer_folds"] >= 3 for row in target_rows
        ),
    }
    report = {
        "status": "CANDIDATE_SELECTED" if all(gates.values()) else "REJECTED",
        "contract": "four-reader target-specific nested blend; no submission",
        "input_path": str(input_path),
        "input_sha256": sha256(input_path),
        "study_count": 58,
        "target_count": 12,
        "arm_names": arm_names.tolist(),
        "anchor_weights": ANCHOR.tolist(),
        "grid_step": GRID_STEP,
        "grid_size": int(len(grid)),
        "l2_penalty": L2_PENALTY,
        "outer_splits": OUTER_SPLITS,
        "nested_coverage_complete": bool((fold_index >= 0).all()),
        "anchor_macro_auc": anchor_auc,
        "nested_candidate_macro_auc": candidate_auc,
        "nested_delta": delta,
        "outer_fold_deltas": outer_deltas,
        "targets": target_rows,
        "final_weights": {
            targets[index]: final_weights[index].tolist() for index in range(12)
        },
        "bootstrap": bootstrap,
        "gates": gates,
        "submission_created": False,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("multireader_nested_report.json"))
    args = parser.parse_args()
    report = run(args.input, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
