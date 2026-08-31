"""Repeated nested source-weight search for public report-derived knee labels."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def rank_full(path: Path, ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    ranked = frame[TARGETS].rank(method="average", pct=True)
    ranked.insert(0, "StudyInstanceUID", frame["StudyInstanceUID"])
    aligned = pd.DataFrame({"StudyInstanceUID": ids}).merge(
        ranked, on="StudyInstanceUID", how="left", validate="one_to_one", sort=False,
    )
    return np.nan_to_num(aligned[TARGETS].to_numpy(np.float64), nan=0.5)


def simplex_weights(source_count: int, units: int = 4) -> np.ndarray:
    values = [parts for parts in product(range(units + 1), repeat=source_count) if sum(parts) == units]
    return np.asarray(values, dtype=np.float64) / units


def macro_auc(y, pred):
    return float(np.mean([roc_auc_score(y[:, j], pred[:, j]) for j in range(y.shape[1])]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--lixin", type=Path, required=True)
    parser.add_argument("--steven-full", type=Path, required=True)
    parser.add_argument("--steven-v2", type=Path, required=True)
    parser.add_argument("--steven-v4", type=Path, required=True)
    parser.add_argument("--pilkwang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    gold = train.loc[train[TARGETS].notna().all(axis=1), ["StudyInstanceUID", *TARGETS]]
    ids = gold["StudyInstanceUID"].tolist()
    y = gold[TARGETS].to_numpy(np.int64)
    source_paths = {
        "lixin_sol56": args.lixin,
        "steven_full": args.steven_full,
        "steven_v2": args.steven_v2,
        "steven_v4": args.steven_v4,
        "pilkwang_v2": args.pilkwang,
    }
    names = list(source_paths)
    source = np.stack([rank_full(path, ids) for path in source_paths.values()])
    baseline = source[names.index("steven_v4")]
    baseline_score = macro_auc(y, baseline)
    grid = simplex_weights(len(names), units=4)
    repeat_deltas = []
    chosen = []
    for repeat in range(args.repeats):
        crossfit = np.zeros_like(baseline)
        repeat_weights = []
        for target in range(len(TARGETS)):
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260831 + repeat)
            target_weights = []
            for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y[:, target]):
                predictions = np.tensordot(grid, source[:, train_idx, target], axes=(1, 0))
                aucs = np.asarray([roc_auc_score(y[train_idx, target], row) for row in predictions])
                # Prefer the simplest source-near solution when training AUC ties.
                best_score = aucs.max()
                tied = np.flatnonzero(np.isclose(aucs, best_score, atol=1e-12))
                best = tied[np.argmax(grid[tied, names.index("steven_v4")])]
                weight = grid[best]
                crossfit[valid_idx, target] = weight @ source[:, valid_idx, target]
                target_weights.append(weight)
            repeat_weights.append(np.median(np.stack(target_weights), axis=0))
        score = macro_auc(y, crossfit)
        repeat_deltas.append(score - baseline_score)
        chosen.append(np.stack(repeat_weights))

    deltas = np.asarray(repeat_deltas)
    weights = np.stack(chosen)
    report = {
        "protocol": "100 repeated target-wise nested stratified five-fold searches; 0.25 simplex; no fold sees validation labels",
        "sources": names,
        "baseline": "steven_v4",
        "baseline_macro_auc": baseline_score,
        "repeats": args.repeats,
        "nested_delta": {
            "mean": float(deltas.mean()), "median": float(np.median(deltas)),
            "p_improve": float(np.mean(deltas > 0)),
            "q05": float(np.quantile(deltas, 0.05)), "q95": float(np.quantile(deltas, 0.95)),
        },
        "median_weights_by_target": {
            target: {name: float(value) for name, value in zip(names, np.median(weights[:, index], axis=0))}
            for index, target in enumerate(TARGETS)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
