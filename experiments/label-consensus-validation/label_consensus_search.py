"""Leakage-aware comparison of public report-derived label sets.

Only fixed, predeclared aggregation rules are evaluated. The 58 organizer labels
are used for final measurement, never to fit source weights or thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def align(path: Path, ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"duplicate study IDs in {path}")
    aligned = pd.DataFrame({"StudyInstanceUID": ids}).merge(
        frame[["StudyInstanceUID", *TARGETS]],
        on="StudyInstanceUID", how="left", validate="one_to_one", sort=False,
    )
    return aligned


def rank_full(path: Path, ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    ranked = frame[TARGETS].rank(method="average", pct=True)
    ranked.insert(0, "StudyInstanceUID", frame["StudyInstanceUID"])
    aligned = pd.DataFrame({"StudyInstanceUID": ids}).merge(
        ranked, on="StudyInstanceUID", how="left", validate="one_to_one", sort=False,
    )
    return aligned[TARGETS].to_numpy(np.float64)


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    per_target = {
        target: float(roc_auc_score(y[:, index], pred[:, index]))
        for index, target in enumerate(TARGETS)
    }
    return {"macro_auc": float(np.mean(list(per_target.values()))), "per_target": per_target}


def bootstrap_delta(y, reference, candidate, draws=500, seed=20260831):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        usable = [j for j in range(y.shape[1]) if np.unique(y[idx, j]).size == 2]
        if not usable:
            continue
        ref = np.mean([roc_auc_score(y[idx, j], reference[idx, j]) for j in usable])
        cand = np.mean([roc_auc_score(y[idx, j], candidate[idx, j]) for j in usable])
        values.append(cand - ref)
    values = np.asarray(values)
    return {
        "draws": int(len(values)), "mean": float(values.mean()),
        "p_improve": float(np.mean(values > 0)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--lixin", type=Path, required=True)
    parser.add_argument("--steven-full", type=Path, required=True)
    parser.add_argument("--steven-v2", type=Path, required=True)
    parser.add_argument("--steven-v4", type=Path, required=True)
    parser.add_argument("--pilkwang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    gold_mask = train[TARGETS].notna().all(axis=1)
    gold = train.loc[gold_mask, ["StudyInstanceUID", *TARGETS]].copy()
    if len(gold) != 58:
        raise ValueError(f"expected 58 gold rows, found {len(gold)}")
    ids = gold["StudyInstanceUID"].tolist()
    y = gold[TARGETS].to_numpy(np.int64)

    paths = {
        "lixin_sol56": args.lixin,
        "steven_full": args.steven_full,
        "steven_v2": args.steven_v2,
        "steven_v4": args.steven_v4,
        "pilkwang_v2": args.pilkwang,
    }
    ranked = {name: rank_full(path, ids) for name, path in paths.items()}
    missing = {name: int(np.isnan(value).sum()) for name, value in ranked.items()}
    # Pilkwang has one absent study. A neutral rank is used only for that missing cell.
    ranked = {name: np.nan_to_num(value, nan=0.5) for name, value in ranked.items()}

    candidates = dict(ranked)
    candidates["fixed_three_source_rank_mean"] = np.mean(
        [ranked["lixin_sol56"], ranked["steven_v4"], ranked["pilkwang_v2"]], axis=0
    )
    candidates["fixed_three_source_rank_median"] = np.median(
        np.stack([ranked["lixin_sol56"], ranked["steven_v4"], ranked["pilkwang_v2"]]), axis=0
    )
    candidates["fixed_all_source_rank_mean"] = np.mean(list(ranked.values()), axis=0)
    # This rule treats source disagreement as uncertainty without learning from gold labels.
    stack = np.stack([ranked["lixin_sol56"], ranked["steven_v4"], ranked["pilkwang_v2"]])
    agreement = stack.max(axis=0) - stack.min(axis=0) <= 0.20
    candidates["agreement_gated_three_source"] = np.where(agreement, stack.mean(axis=0), 0.5)

    evaluated = {name: scores(y, pred) for name, pred in candidates.items()}
    best_source = max(paths, key=lambda name: evaluated[name]["macro_auc"])
    best_fixed = max(
        (name for name in candidates if name not in paths),
        key=lambda name: evaluated[name]["macro_auc"],
    )
    report = {
        "protocol": "fixed aggregation rules; no gold-fitted weights or thresholds",
        "gold_rows": len(gold), "missing_values_after_alignment": missing,
        "results": evaluated, "best_individual_source": best_source,
        "best_fixed_consensus": best_fixed,
        "best_fixed_delta": evaluated[best_fixed]["macro_auc"] - evaluated[best_source]["macro_auc"],
        "best_fixed_bootstrap": bootstrap_delta(y, candidates[best_source], candidates[best_fixed]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
