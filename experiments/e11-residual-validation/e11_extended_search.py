"""Extend the E11 residual ladder after the original 0.35 ceiling was selected."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oof_search import (
    TARGETS,
    aucs,
    bootstrap_delta,
    load_csv_arm,
    nested_residual_search,
    rank_columns,
    repeated_nested_residual,
)


SUPPORTED = {"ACL", "Medial OA", "Lateral OA", "PF OA", "Synovitis", "Contusion"}


def align_npz(ids: np.ndarray, values: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    lookup = {str(uid): i for i, uid in enumerate(ids.astype(str))}
    return np.asarray([values[lookup[str(uid)]] for uid in wanted], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(
        args.assets / "competition" / "train.csv",
        dtype={"StudyInstanceUID": str},
    )
    gold = train[TARGETS].notna().all(axis=1)
    ids = train.loc[gold, "StudyInstanceUID"].astype(str).to_numpy()
    y = train.loc[gold, TARGETS].to_numpy(np.int64)

    dino_npz = np.load(args.assets / "dino-oof" / "oof.npz", allow_pickle=False)
    dino = rank_columns(align_npz(dino_npz["ids"], dino_npz["pred"], ids))
    rad_e9_raw, folds = load_csv_arm(args.assets / "rad-e9" / "v52_oof.csv", ids)
    public_e11_raw, public_folds = load_csv_arm(
        args.assets / "rad-e11" / "v52_e11_oof.csv", ids
    )
    remote_e11_raw, remote_folds = load_csv_arm(
        args.assets / "e11" / "v52_e11_oof.csv", ids
    )
    remote_e13_raw, remote_e13_folds = load_csv_arm(
        args.assets / "e13" / "v52_e11_oof.csv", ids
    )
    for other in (public_folds, remote_folds, remote_e13_folds):
        if not np.array_equal(folds, other):
            raise ValueError("OOF fold assignments disagree")

    rad_e9 = rank_columns(rad_e9_raw)
    public_e11 = rank_columns(public_e11_raw)
    remote_e11 = rank_columns(remote_e11_raw)
    remote_e13 = rank_columns(remote_e13_raw)
    reference = rank_columns(0.50 * rad_e9 + 0.50 * remote_e13)
    e10 = dino.copy()
    for j, target in enumerate(TARGETS):
        if target not in {"Baker's", "Fracture"}:
            e10[:, j] = 0.50 * dino[:, j] + 0.50 * reference[:, j]
    e10 = rank_columns(e10)
    base = rank_columns(0.85 * e10 + 0.15 * remote_e11)

    ladder = np.asarray(
        [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.60, 0.70, 0.80]
    )
    crossfit, choices = nested_residual_search(y, base, public_e11, folds, ladder)
    outer = np.asarray([choice["alphas"] for choice in choices], dtype=np.float64)
    all_median = np.median(outer, axis=0)
    supported_alpha = np.asarray(
        [all_median[j] if target in SUPPORTED else 0.0 for j, target in enumerate(TARGETS)]
    )
    fixed = rank_columns(
        (1.0 - supported_alpha[None, :]) * base
        + supported_alpha[None, :] * public_e11
    )
    base_score = float(aucs(y, base).mean())
    crossfit_score = float(aucs(y, crossfit).mean())
    fixed_score = float(aucs(y, fixed).mean())
    repeated = repeated_nested_residual(y, base, public_e11, ladder, repeats=100)
    report = {
        "supported_targets": sorted(SUPPORTED),
        "ladder": ladder.tolist(),
        "base_macro_auc": base_score,
        "nested_macro_auc": crossfit_score,
        "nested_delta": crossfit_score - base_score,
        "nested_bootstrap": bootstrap_delta(y, base, crossfit, draws=1000),
        "outer_choices": choices,
        "median_alpha_all_targets": dict(zip(TARGETS, all_median)),
        "supported_fixed_alphas": dict(zip(TARGETS, supported_alpha)),
        "supported_fixed_macro_auc": fixed_score,
        "supported_fixed_delta": fixed_score - base_score,
        "supported_fixed_bootstrap": bootstrap_delta(y, base, fixed, draws=1000),
        "repeated_random_outer_folds": repeated,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
