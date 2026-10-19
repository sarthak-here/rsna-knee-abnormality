"""Read-only audit of existing artifacts; no fitting or candidate selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_auc(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean([
        roc_auc_score(truth[:, target], prediction[:, target])
        for target in range(12)
    ]))


def audit() -> dict:
    cache = ROOT / ".audit/multireader-v3/multireader_gold_predictions.npz"
    router_cache = ROOT / "experiments/uncertainty-router/output-v4/uncertainty_router_gold_cache.npz"
    manifest_path = ROOT / ".audit/r6-w1/manifest.json"
    with np.load(cache, allow_pickle=False) as package:
        ids = package["ids"].astype(str)
        truth = package["truth"].astype(np.uint8)
        arms = package["arm_names"].astype(str)
        prediction = package["predictions"].astype(np.float64)
    if prediction.shape != (58, 4, 12) or len(set(ids)) != 58:
        raise RuntimeError("audit input shape/identity drift")
    if arms.tolist() != ["r6_flip", "raptor384", "raptor224", "r7"]:
        raise RuntimeError("audit arm identity drift")
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite audit input")
    weights = np.asarray([.5, .325, .175, 0.])
    ranked = np.stack([
        pd.DataFrame(prediction[:, arm]).rank(method="average", pct=True).to_numpy()
        for arm in range(4)
    ], axis=1)
    raw_anchor = np.einsum("nat,a->nt", prediction, weights)
    exact_rank_anchor = np.einsum("nat,a->nt", ranked, weights)
    # The old nested optimizer uses ordinal rather than average-tie ranks.
    ordinal = np.empty_like(prediction)
    for arm in range(4):
        for target in range(12):
            order = np.argsort(prediction[:, arm, target], kind="mergesort")
            ordinal[order, arm, target] = np.arange(58) / 57
    ordinal_anchor = np.einsum("nat,a->nt", ordinal, weights)
    manifest = json.loads(manifest_path.read_text())
    result = {
        "purpose": "existing-model audit only; no training, search, or submission",
        "results_documents": len(list((ROOT / "experiments").rglob("RESULTS.md"))),
        "multireader_cache_sha256": sha256(cache),
        "individual_gold_auc": {
            str(arm): macro_auc(truth, prediction[:, index])
            for index, arm in enumerate(arms)
        },
        "same_cache_anchors": {
            "raw_probability_blend_auc": macro_auc(truth, raw_anchor),
            "source_faithful_average_rank_blend_auc": macro_auc(truth, exact_rank_anchor),
            "old_nested_ordinal_rank_blend_auc": macro_auc(truth, ordinal_anchor),
        },
        "r6_manifest_sha256": sha256(manifest_path),
        "r6_training_recipe": manifest["run_config"],
        "r6_pool_count": manifest["run_config"]["pool_studies"],
        "r6_target_path": manifest["run_config"]["targets"],
        "r6_manifest_gold_in_training": manifest["gold_in_training"],
        "required_but_unproven": [
            "base-model exclusion of every outer validation group",
            "teacher exclusion of the entire outer validation group",
            "complete original R6 targets, state targets, pool IDs, and training cache",
            "a measured training budget for the exact R6 recipe",
        ],
    }
    with np.load(router_cache, allow_pickle=False) as package:
        router_truth = package["truth"].astype(np.uint8)
        logits = package["r6_logits"].astype(np.float64)
        deployed = package["r6_prob"].astype(np.float64)
        head = 1.0 / (1.0 + np.exp(-np.clip(logits.mean(axis=(1, 2)), -30, 30)))
        result["r6_stackers_comparison"] = {
            "cache_sha256": sha256(router_cache),
            "deployed_r6_auc": macro_auc(router_truth, deployed),
            "equal_fold_pass_head_auc": macro_auc(router_truth, head),
            "differing_probability_cells": int((np.abs(deployed - head) > 1e-6).sum()),
            "max_abs_probability_difference": float(np.abs(deployed - head).max()),
        }
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
