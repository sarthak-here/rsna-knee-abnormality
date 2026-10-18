import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "cache" / "multireader_gold_predictions.npz"
EXPECTED_SHA = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"
ARMS = ["r6_flip", "raptor384", "raptor224", "r7"]
ANCHOR_WEIGHTS = np.asarray([0.50, 0.325, 0.175, 0.0], np.float64)
MAX_R7_SLOT = 0.10
TEMPERATURE = 0.10


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values):
    out = np.empty_like(values, dtype=np.float64)
    for j in range(values.shape[1]):
        order = np.argsort(values[:, j], kind="mergesort")
        sorted_values = values[order, j]
        ranked = np.empty(len(values), np.float64)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranked[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        out[:, j] = (ranked + 0.5) / len(values)
    return out


def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12)]))


def target_aucs(y, p):
    return np.asarray([roc_auc_score(y[:, j], p[:, j]) for j in range(12)])


def paired_bootstrap(y, anchor, candidate, draws=10000, seed=20261018):
    def fast_auc(labels, scores):
        positive = scores[labels == 1]
        negative = scores[labels == 0]
        comparisons = positive[:, None] - negative[None, :]
        return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) /
                     comparisons.size)

    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(draws):
        ix = rng.integers(0, len(y), len(y))
        scores = []
        for prediction in (anchor, candidate):
            valid = [fast_auc(y[ix, j], prediction[ix, j]) for j in range(12)
                     if len(np.unique(y[ix, j])) == 2]
            scores.append(np.mean(valid))
        delta.append(scores[1] - scores[0])
    delta = np.asarray(delta)
    return {"mean": float(delta.mean()), "p_improve": float((delta > 0).mean()),
            "q025": float(np.quantile(delta, 0.025)),
            "q975": float(np.quantile(delta, 0.975))}


if sha256(INPUT) != EXPECTED_SHA:
    raise RuntimeError("immutable prediction cache identity drift")
package = np.load(INPUT, allow_pickle=False)
if package["ids"].shape != (58,) or len(set(package["ids"].astype(str))) != 58:
    raise RuntimeError("ID integrity failure")
if package["arm_names"].astype(str).tolist() != ARMS:
    raise RuntimeError("reader identity/order drift")
truth = package["truth"].astype(np.uint8)
prediction = package["predictions"].astype(np.float64)
if truth.shape != (58, 12) or prediction.shape != (58, 4, 12):
    raise RuntimeError("tensor shape drift")
if not np.isin(truth, [0, 1]).all() or not np.isfinite(prediction).all():
    raise RuntimeError("value integrity failure")
if prediction.min() < 0 or prediction.max() > 1:
    raise RuntimeError("probability range failure")

ranked = np.stack([ranks(prediction[:, arm, :]) for arm in range(4)], axis=1)
anchor = np.einsum("nat,a->nt", ranked, ANCHOR_WEIGHTS)
consensus = np.median(ranked[:, :3, :], axis=1)
agreement = np.exp(-np.abs(ranked[:, 3, :] - consensus) / TEMPERATURE)
slot = MAX_R7_SLOT * agreement
candidate = (1.0 - slot) * anchor + slot * ranked[:, 3, :]

anchor_auc = macro_auc(truth, anchor)
candidate_auc = macro_auc(truth, candidate)
target_delta = target_aucs(truth, candidate) - target_aucs(truth, anchor)
bootstrap = paired_bootstrap(truth, anchor, candidate)
gate = {
    "macro_gain_at_least_0_001": candidate_auc - anchor_auc >= 0.001,
    "nonnegative_targets_at_least_8": int((target_delta >= 0).sum()) >= 8,
    "bootstrap_p_at_least_0_90": bootstrap["p_improve"] >= 0.90,
    "bootstrap_q025_above_minus_0_001": bootstrap["q025"] > -0.001,
    "integrity": True,
}
report = {
    "contract": "label-free consensus-gated R7 residual; no submission",
    "prediction_cache_sha256": EXPECTED_SHA,
    "anchor_weights": ANCHOR_WEIGHTS.tolist(),
    "max_r7_slot": MAX_R7_SLOT,
    "temperature": TEMPERATURE,
    "anchor_auc": anchor_auc,
    "candidate_auc": candidate_auc,
    "candidate_minus_anchor": candidate_auc - anchor_auc,
    "nonnegative_targets": int((target_delta >= 0).sum()),
    "mean_r7_slot": float(slot.mean()),
    "per_target_delta": target_delta.tolist(),
    "bootstrap": bootstrap,
    "gate": gate,
    "passed": bool(all(gate.values())),
    "submission_created": False,
}
(ROOT / "consensus_gate_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
