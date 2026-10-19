"""Predeclared CPU evaluation for target-wise R6 fold/pass calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
OOF_SHA256 = "60529a0ba00107abd7efd101f70454e7ce89407befbaa4107f24771ee088579e"
ALPHA = 100.0
EPS = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, EPS, 1.0 - EPS)
    return np.log(values / (1.0 - values))


def target_aucs(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    scores = []
    for target in range(truth.shape[1]):
        if np.unique(truth[:, target]).size != 2:
            raise RuntimeError(f"target {target} lacks both classes")
        scores.append(roc_auc_score(truth[:, target], prediction[:, target]))
    return np.asarray(scores, np.float64)


def macro_auc(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(target_aucs(truth, prediction).mean())


def bootstrap(truth: np.ndarray, control: np.ndarray, candidate: np.ndarray,
              draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        rows = rng.integers(0, len(truth), len(truth))
        try:
            deltas.append(macro_auc(truth[rows], candidate[rows]) -
                          macro_auc(truth[rows], control[rows]))
        except RuntimeError:
            continue
    values = np.asarray(deltas, np.float64)
    if values.size < int(draws * .95):
        raise RuntimeError("too many invalid bootstrap draws")
    return {"draws": int(values.size), "mean": float(values.mean()),
            "p_improve": float((values > 0).mean()),
            "q025": float(np.quantile(values, .025)),
            "q975": float(np.quantile(values, .975))}


def fit_predict(x_fit: np.ndarray, y_fit: np.ndarray,
                x_eval: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-6] = 1.0
    model = Ridge(alpha=ALPHA, fit_intercept=True)
    model.fit((x_fit - mean) / scale, logit(y_fit))
    prediction = sigmoid(model.predict((x_eval - mean) / scale))
    receipt = {"mean": mean.tolist(), "scale": scale.tolist(),
               "coef": model.coef_.tolist(),
               "intercept": float(model.intercept_), "alpha": ALPHA}
    return prediction.astype(np.float32), receipt


def evaluate(cache_path: Path, train_path: Path, labels_path: Path,
             teacher_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in output_dir.glob("*.csv"):
        csv_path.unlink()

    train = pd.read_csv(train_path, dtype={"StudyInstanceUID": str})
    ids = train.StudyInstanceUID.astype(str).to_numpy()
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if ids.shape != (4407,) or len(set(ids.tolist())) != 4407 or int(gold.sum()) != 58:
        raise RuntimeError("official identity/count drift")
    report_column = next((column for column in train.columns
                          if column.lower() in {"report", "report_text", "findings"}), None)
    if report_column is None:
        raise RuntimeError("radiology report column not found")

    with np.load(cache_path, allow_pickle=False) as package:
        if package["ids"].astype(str).tolist() != ids.tolist():
            raise RuntimeError("R6 cache ordering drift")
        if package["targets"].astype(str).tolist() != TARGETS:
            raise RuntimeError("R6 target ordering drift")
        probability = package["probability"].astype(np.float32)
        logits = package["logits"].astype(np.float32)
    if probability.shape != (4407, 12) or logits.shape != (4407, 4, 2, 12):
        raise RuntimeError("R6 cache shape drift")
    if not np.isfinite(probability).all() or not np.isfinite(logits).all():
        raise RuntimeError("non-finite R6 cache")
    if probability.min() < 0 or probability.max() > 1:
        raise RuntimeError("R6 probability range drift")

    labels = pd.read_csv(labels_path, dtype={"StudyInstanceUID": str})
    labels = train[["StudyInstanceUID"]].merge(
        labels, on="StudyInstanceUID", how="left", validate="one_to_one")
    steven = labels[TARGETS].to_numpy(np.float32)
    if not np.isfinite(steven).all():
        raise RuntimeError("Steven-v4 label coverage drift")

    if sha256(teacher_path) != OOF_SHA256:
        raise RuntimeError("teacher prediction identity drift")
    teacher = pd.read_csv(teacher_path, dtype={"StudyInstanceUID": str})
    if teacher.columns.tolist() != ["StudyInstanceUID", "teacher_fold"] + TARGETS:
        raise RuntimeError("teacher schema drift")
    if len(teacher) != 2610 or teacher.StudyInstanceUID.nunique() != 2610:
        raise RuntimeError("teacher coverage drift")
    if set(teacher.StudyInstanceUID) & set(ids[gold]):
        raise RuntimeError("organizer gold leaked into teacher predictions")
    aligned_teacher = train[["StudyInstanceUID"]].merge(
        teacher, on="StudyInstanceUID", how="left", validate="one_to_one")
    covered = aligned_teacher[TARGETS].notna().all(axis=1).to_numpy() & ~gold
    corrected = steven.copy()
    teacher_values = aligned_teacher[TARGETS].to_numpy(np.float32)
    corrected[covered] = .5 * steven[covered] + .5 * teacher_values[covered]
    fit_rows = np.flatnonzero(~gold)
    gold_rows = np.flatnonzero(gold)
    if len(fit_rows) != 4349 or int(covered.sum()) != 2610:
        raise RuntimeError("corrected target count drift")

    groups = train.loc[~gold, report_column].fillna("").astype(str).map(
        lambda value: hashlib.sha256(value.strip().lower().encode()).hexdigest()
    ).to_numpy()
    fold = np.full(4349, -1, np.int8)
    for fold_index, (_, heldout) in enumerate(
        GroupKFold(n_splits=5).split(fit_rows, groups=groups)
    ):
        fold[heldout] = fold_index
    if (fold < 0).any():
        raise RuntimeError("report-group fold coverage drift")

    features = logits.transpose(0, 3, 1, 2).reshape(4407, 12, 8)
    baseline = sigmoid(logits.mean(axis=(1, 2)))
    crossfit = np.empty((4349, 12), np.float32)
    fold_receipts = []
    fold_deltas = []
    hard = (corrected[fit_rows] >= .5).astype(np.uint8)
    for fold_index in range(5):
        heldout_local = np.flatnonzero(fold == fold_index)
        fitting_local = np.flatnonzero(fold != fold_index)
        target_receipts = []
        for target in range(12):
            values, receipt = fit_predict(
                features[fit_rows[fitting_local], target],
                corrected[fit_rows[fitting_local], target],
                features[fit_rows[heldout_local], target],
            )
            crossfit[heldout_local, target] = values
            target_receipts.append(receipt)
        delta = (macro_auc(hard[heldout_local], crossfit[heldout_local]) -
                 macro_auc(hard[heldout_local], baseline[fit_rows[heldout_local]]))
        fold_deltas.append(float(delta))
        fold_receipts.append({"fold": fold_index, "heldout": int(len(heldout_local)),
                              "delta": float(delta), "targets": target_receipts})

    oof_control = baseline[fit_rows]
    oof_control_auc = macro_auc(hard, oof_control)
    oof_candidate_auc = macro_auc(hard, crossfit)
    oof_target_delta = target_aucs(hard, crossfit) - target_aucs(hard, oof_control)
    oof_bootstrap = bootstrap(hard, oof_control, crossfit, 5000, 20261019)

    gold_prediction = np.empty((58, 12), np.float32)
    final_receipts = []
    for target in range(12):
        values, receipt = fit_predict(features[fit_rows, target], corrected[fit_rows, target],
                                      features[gold_rows, target])
        gold_prediction[:, target] = values
        final_receipts.append(receipt)
    gold_truth = train.loc[gold, TARGETS].to_numpy(np.uint8)
    gold_control = baseline[gold_rows]
    gold_control_auc = macro_auc(gold_truth, gold_control)
    gold_candidate_auc = macro_auc(gold_truth, gold_prediction)
    gold_target_delta = target_aucs(gold_truth, gold_prediction) - target_aucs(gold_truth, gold_control)
    gold_bootstrap = bootstrap(gold_truth, gold_control, gold_prediction, 10000, 20261020)

    gates = {
        "oof_gain_at_least_0_003": oof_candidate_auc - oof_control_auc >= .003,
        "oof_nonnegative_targets_at_least_8": int((oof_target_delta >= 0).sum()) >= 8,
        "positive_report_folds_at_least_4": int((np.asarray(fold_deltas) > 0).sum()) >= 4,
        "oof_bootstrap_p_at_least_0_99": oof_bootstrap["p_improve"] >= .99,
        "oof_bootstrap_q025_positive": oof_bootstrap["q025"] > 0,
        "gold_gain_at_least_0_001": gold_candidate_auc - gold_control_auc >= .001,
        "gold_nonnegative_targets_at_least_8": int((gold_target_delta >= 0).sum()) >= 8,
        "gold_bootstrap_p_at_least_0_90": gold_bootstrap["p_improve"] >= .90,
        "gold_bootstrap_q025_above_minus_0_001": gold_bootstrap["q025"] > -.001,
    }
    report = {
        "contract": "fixed target-wise eight-logit R6 ridge calibration; no submission",
        "cache_sha256": sha256(cache_path), "teacher_sha256": OOF_SHA256,
        "alpha": ALPHA, "fit_studies": 4349, "corrected": 2610,
        "unchanged": 1739, "gold_fit_count": 0, "gold_evaluation_count": 58,
        "oof": {"control_auc": oof_control_auc, "candidate_auc": oof_candidate_auc,
                "gain": oof_candidate_auc - oof_control_auc,
                "target_delta": oof_target_delta.tolist(), "fold_delta": fold_deltas,
                "bootstrap": oof_bootstrap},
        "gold": {"control_auc": gold_control_auc, "candidate_auc": gold_candidate_auc,
                 "gain": gold_candidate_auc - gold_control_auc,
                 "target_delta": gold_target_delta.tolist(), "bootstrap": gold_bootstrap},
        "fold_receipts": fold_receipts, "final_receipts": final_receipts,
        "gates": {key: bool(value) for key, value in gates.items()},
        "submission_created": False,
    }
    report["passed"] = bool(all(report["gates"].values()))
    report_path = output_dir / "r6_crossfit_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output_dir / "r6_crossfit_calibration_predictions.npz",
                        ids=ids[gold_rows].astype("U80"), truth=gold_truth,
                        control=gold_control, candidate=gold_prediction,
                        targets=np.asarray(TARGETS, dtype="U32"))
    if list(output_dir.glob("*.csv")):
        raise RuntimeError("forbidden CSV artifact")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.cache, args.train, args.labels, args.teacher,
                              args.output_dir), indent=2))


if __name__ == "__main__":
    main()
