"""Offline OOF ensemble search for the RSNA knee competition."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def rank_columns(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(method="average", pct=True).to_numpy(np.float64)


def aucs(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.asarray(
        [roc_auc_score(y[:, j], pred[:, j]) for j in range(y.shape[1])],
        dtype=np.float64,
    )


def load_csv_arm(path: Path, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"duplicate IDs in {path}")
    aligned = pd.DataFrame({"StudyInstanceUID": ids}).merge(
        frame[["StudyInstanceUID", *TARGETS, "fold", "is_gold"]],
        on="StudyInstanceUID",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned[TARGETS].isna().any().any():
        raise ValueError(f"missing predictions after alignment: {path}")
    return aligned[TARGETS].to_numpy(np.float64), aligned["fold"].to_numpy(int)


def simplex_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    units = int(round(1.0 / step))
    return [
        (a / units, b / units, (units - a - b) / units)
        for a in range(units + 1)
        for b in range(units + 1 - a)
    ]


def blend(arms: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    result = sum(weight * arm for weight, arm in zip(weights, arms, strict=True))
    return rank_columns(result)


def nested_global_search(
    y: np.ndarray,
    arms: list[np.ndarray],
    folds: np.ndarray,
    grid: list[tuple[float, float, float]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    crossfit = np.zeros_like(y, dtype=np.float64)
    choices: list[dict[str, object]] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = ~train
        scored = []
        for weights in grid:
            score = float(aucs(y[train], blend([arm[train] for arm in arms], weights)).mean())
            scored.append((score, weights))
        score, weights = max(scored, key=lambda item: (item[0], -item[1][2]))
        raw = sum(weight * arm[valid] for weight, arm in zip(weights, arms, strict=True))
        crossfit[valid] = raw
        choices.append({"fold": int(fold), "train_macro_auc": score, "weights": weights})
    return rank_columns(crossfit), choices


def nested_residual_search(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    folds: np.ndarray,
    ladder: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    crossfit = np.zeros_like(y, dtype=np.float64)
    choices: list[dict[str, object]] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = ~train
        selected = []
        for target in range(y.shape[1]):
            scores = []
            for alpha in ladder:
                pred = rank_columns(
                    (1.0 - alpha) * base[train, [target]]
                    + alpha * candidate[train, [target]]
                )[:, 0]
                scores.append(roc_auc_score(y[train, target], pred))
            best = int(np.argmax(scores))
            alpha = float(ladder[best])
            crossfit[valid, target] = (
                (1.0 - alpha) * base[valid, target]
                + alpha * candidate[valid, target]
            )
            selected.append(alpha)
        choices.append({"fold": int(fold), "alphas": selected})
    return rank_columns(crossfit), choices


def bootstrap_delta(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    draws: int = 1000,
    seed: int = 20260831,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        usable = [j for j in range(y.shape[1]) if np.unique(y[idx, j]).size == 2]
        if not usable:
            continue
        base_auc = np.mean([roc_auc_score(y[idx, j], base[idx, j]) for j in usable])
        cand_auc = np.mean([roc_auc_score(y[idx, j], candidate[idx, j]) for j in usable])
        deltas.append(cand_auc - base_auc)
    values = np.asarray(deltas)
    return {
        "draws": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p_improve": float(np.mean(values > 0)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def repeated_nested_residual(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    ladder: np.ndarray,
    repeats: int = 100,
) -> dict[str, object]:
    base_score = float(aucs(y, base).mean())
    deltas = []
    alpha_records = []
    for repeat in range(repeats):
        splitter = KFold(n_splits=5, shuffle=True, random_state=20260831 + repeat)
        folds = np.empty(len(y), dtype=int)
        for fold, (_, valid) in enumerate(splitter.split(y)):
            folds[valid] = fold
        crossfit, choices = nested_residual_search(y, base, candidate, folds, ladder)
        deltas.append(float(aucs(y, crossfit).mean()) - base_score)
        alpha_records.extend(choice["alphas"] for choice in choices)
    values = np.asarray(deltas)
    alpha_values = np.asarray(alpha_records, dtype=np.float64)
    return {
        "repeats": repeats,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "p_improve": float(np.mean(values > 0)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "median_alpha_by_target": dict(zip(TARGETS, np.median(alpha_values, axis=0))),
        "q25_alpha_by_target": dict(zip(TARGETS, np.quantile(alpha_values, 0.25, axis=0))),
        "selection_rate_by_target": dict(zip(TARGETS, np.mean(alpha_values > 0, axis=0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.assets / "competition" / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1)
    gold_ids = train.loc[gold, "StudyInstanceUID"].astype(str).to_numpy()
    y = train.loc[gold, TARGETS].to_numpy(np.int64)
    if len(gold_ids) != 58:
        raise ValueError(f"expected 58 gold studies, found {len(gold_ids)}")

    dino = np.load(args.assets / "dino-oof" / "oof.npz", allow_pickle=False)
    dino_ids = dino["ids"].astype(str)
    dino_frame = pd.DataFrame(dino["pred"], columns=TARGETS)
    dino_frame.insert(0, "StudyInstanceUID", dino_ids)
    dino_aligned = pd.DataFrame({"StudyInstanceUID": gold_ids}).merge(
        dino_frame, on="StudyInstanceUID", how="left", validate="one_to_one", sort=False
    )
    if dino_aligned[TARGETS].isna().any().any():
        raise ValueError("DINO OOF alignment failed")

    rad_e9, folds = load_csv_arm(args.assets / "rad-e9" / "v52_oof.csv", gold_ids)
    rad_e11, folds_e11 = load_csv_arm(args.assets / "rad-e11" / "v52_e11_oof.csv", gold_ids)
    if not np.array_equal(folds, folds_e11):
        raise ValueError("RadImageNet fold assignments disagree")

    arms = {
        "dino": rank_columns(dino_aligned[TARGETS].to_numpy(np.float64)),
        "rad_e9": rank_columns(rad_e9),
        "rad_e11": rank_columns(rad_e11),
    }
    individual = {
        name: {"macro_auc": float(aucs(y, pred).mean()), "per_target": dict(zip(TARGETS, aucs(y, pred)))}
        for name, pred in arms.items()
    }

    grid = simplex_grid(0.05)
    full_grid = []
    arm_list = [arms["dino"], arms["rad_e9"], arms["rad_e11"]]
    for weights in grid:
        pred = blend(arm_list, weights)
        full_grid.append((float(aucs(y, pred).mean()), weights, pred))
    full_score, full_weights, full_pred = max(full_grid, key=lambda item: item[0])

    crossfit_global, global_choices = nested_global_search(y, arm_list, folds, grid)
    base = blend(arm_list, (0.50, 0.50, 0.0))
    ladder = np.asarray([0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.35])
    crossfit_residual, residual_choices = nested_residual_search(
        y, base, arms["rad_e11"], folds, ladder
    )

    outer_alphas = np.asarray([choice["alphas"] for choice in residual_choices])
    fixed_alphas = np.median(outer_alphas, axis=0)
    fixed_residual = rank_columns(
        (1.0 - fixed_alphas[None, :]) * base
        + fixed_alphas[None, :] * arms["rad_e11"]
    )

    base_auc = aucs(y, base)
    global_auc = aucs(y, crossfit_global)
    residual_auc = aucs(y, crossfit_residual)
    report = {
        "gold_rows": len(gold_ids),
        "targets": TARGETS,
        "individual": individual,
        "base_dino_rad_e9_equal": {
            "macro_auc": float(base_auc.mean()),
            "per_target": dict(zip(TARGETS, base_auc)),
        },
        "descriptive_full_grid": {
            "macro_auc": full_score,
            "weights_dino_rad_e9_rad_e11": full_weights,
        },
        "nested_global": {
            "macro_auc": float(global_auc.mean()),
            "delta_vs_base": float(global_auc.mean() - base_auc.mean()),
            "choices": global_choices,
            "per_target": dict(zip(TARGETS, global_auc)),
            "bootstrap": bootstrap_delta(y, base, crossfit_global),
        },
        "nested_target_residual": {
            "macro_auc": float(residual_auc.mean()),
            "delta_vs_base": float(residual_auc.mean() - base_auc.mean()),
            "choices": residual_choices,
            "per_target": dict(zip(TARGETS, residual_auc)),
            "bootstrap": bootstrap_delta(y, base, crossfit_residual),
        },
        "fixed_median_residual": {
            "alphas": dict(zip(TARGETS, fixed_alphas)),
            "macro_auc": float(aucs(y, fixed_residual).mean()),
            "delta_vs_base": float(aucs(y, fixed_residual).mean() - base_auc.mean()),
            "per_target": dict(zip(TARGETS, aucs(y, fixed_residual))),
            "bootstrap": bootstrap_delta(y, base, fixed_residual),
        },
        "repeated_random_outer_folds": repeated_nested_residual(
            y, base, arms["rad_e11"], ladder, repeats=100
        ),
    }

    remote_e11_path = args.assets / "e11" / "v52_e11_oof.csv"
    remote_e13_path = args.assets / "e13" / "v52_e11_oof.csv"
    if remote_e11_path.is_file() and remote_e13_path.is_file():
        remote_e11_raw, remote_e11_folds = load_csv_arm(remote_e11_path, gold_ids)
        remote_e13_raw, remote_e13_folds = load_csv_arm(remote_e13_path, gold_ids)
        if not np.array_equal(folds, remote_e11_folds) or not np.array_equal(
            folds, remote_e13_folds
        ):
            raise ValueError("remote RadImageNet fold assignments disagree")
        remote_e11 = rank_columns(remote_e11_raw)
        remote_e13 = rank_columns(remote_e13_raw)
        reference_proxy = rank_columns(0.50 * arms["rad_e9"] + 0.50 * remote_e13)
        e10_proxy = arms["dino"].copy()
        protected = {"Baker's", "Fracture"}
        for j, target in enumerate(TARGETS):
            if target not in protected:
                e10_proxy[:, j] = (
                    0.50 * arms["dino"][:, j] + 0.50 * reference_proxy[:, j]
                )
        e10_proxy = rank_columns(e10_proxy)
        deployed_proxy = rank_columns(0.85 * e10_proxy + 0.15 * remote_e11)

        proxy_crossfit, proxy_choices = nested_residual_search(
            y, deployed_proxy, arms["rad_e11"], folds, ladder
        )
        proxy_outer_alphas = np.asarray([choice["alphas"] for choice in proxy_choices])
        proxy_fixed_alphas = np.median(proxy_outer_alphas, axis=0)
        proxy_fixed = rank_columns(
            (1.0 - proxy_fixed_alphas[None, :]) * deployed_proxy
            + proxy_fixed_alphas[None, :] * arms["rad_e11"]
        )
        proxy_auc = aucs(y, deployed_proxy)
        proxy_crossfit_auc = aucs(y, proxy_crossfit)
        proxy_fixed_auc = aucs(y, proxy_fixed)
        report["deployed_transformer_proxy"] = {
            "definition": "DINO + E9/E13 reference + remote E11 pass-two proxy",
            "base_macro_auc": float(proxy_auc.mean()),
            "nested_macro_auc": float(proxy_crossfit_auc.mean()),
            "nested_delta": float(proxy_crossfit_auc.mean() - proxy_auc.mean()),
            "nested_bootstrap": bootstrap_delta(y, deployed_proxy, proxy_crossfit),
            "fixed_median_alphas": dict(zip(TARGETS, proxy_fixed_alphas)),
            "fixed_macro_auc": float(proxy_fixed_auc.mean()),
            "fixed_delta": float(proxy_fixed_auc.mean() - proxy_auc.mean()),
            "fixed_bootstrap": bootstrap_delta(y, deployed_proxy, proxy_fixed),
            "choices": proxy_choices,
            "repeated_random_outer_folds": repeated_nested_residual(
                y, deployed_proxy, arms["rad_e11"], ladder, repeats=100
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
