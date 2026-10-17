"""Build the fail-closed quantitative morphology specialist notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FEATURE_SOURCE = (ROOT / "morphology_features.py").read_text(encoding="utf-8")

RUNNER = r'''
from pathlib import Path
import hashlib, json
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
SEEDS = (20261127, 20261128, 20261129, 20261130, 20261201)
H5_SHA = "490e5ccc07143699d05a9eb3ef9583e08795f1c2bacc6d9cf2bd0a9268158526"
STUDIES_SHA = "efc63afee30345f94a916f98ba6b30f266fccfe2976b8475fa86653b2c510513"
OOF_SHA = "60529a0ba00107abd7efd101f70454e7ce89407befbaa4107f24771ee088579e"
ANCHOR_SHA = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"
ANCHOR_ARMS = ["r6_flip", "raptor384", "raptor224", "r7"]
ANCHOR_WEIGHTS = np.asarray([.50, .325, .175, 0.0], np.float64)
RESIDUAL_WEIGHT = .10
OUT = Path("/kaggle/working")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def unique_input(name):
    hits = [path for path in Path("/kaggle/input").rglob(name) if path.is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(hits)}: {hits[:8]}")
    return hits[0]

def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12)]))

def target_aucs(y, p):
    return np.asarray([roc_auc_score(y[:, j], p[:, j]) for j in range(12)])

def ranks(p):
    out = np.empty_like(p, dtype=np.float64)
    for j in range(p.shape[1]):
        order = np.argsort(p[:, j], kind="mergesort")
        values = p[order, j]
        ranked = np.empty(len(p), np.float64)
        start = 0
        while start < len(p):
            stop = start + 1
            while stop < len(p) and values[stop] == values[start]:
                stop += 1
            ranked[order[start:stop]] = .5 * (start + stop - 1)
            start = stop
        out[:, j] = (ranked + .5) / len(p)
    return out

def paired_bootstrap(y, control, candidate, draws=10000, seed=20261202):
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(draws):
        ix = rng.integers(0, len(y), len(y))
        scores = []
        for pred in (control, candidate):
            valid = [roc_auc_score(y[ix, j], pred[ix, j]) for j in range(12)
                     if len(np.unique(y[ix, j])) == 2]
            scores.append(np.mean(valid))
        delta.append(scores[1] - scores[0])
    delta = np.asarray(delta)
    return {"mean": float(delta.mean()), "p_improve": float((delta > 0).mean()),
            "q025": float(np.quantile(delta, .025)),
            "q975": float(np.quantile(delta, .975))}

h5_path = unique_input("train_288x12.h5")
studies_path = unique_input("highres_cache_studies.csv")
oof_path = unique_input("convnext_oof_predictions.csv")
anchor_path = unique_input("multireader_gold_predictions.npz")
if h5_path.stat().st_size != 17051786047 or sha256(h5_path) != H5_SHA:
    raise RuntimeError("physical cache identity drift")
if sha256(studies_path) != STUDIES_SHA or sha256(oof_path) != OOF_SHA:
    raise RuntimeError("manifest or OOF identity drift")
if sha256(anchor_path) != ANCHOR_SHA:
    raise RuntimeError("anchor identity drift")

train = pd.read_csv(unique_input("train.csv"), dtype={"StudyInstanceUID": str})
ids = train.StudyInstanceUID.astype(str).to_numpy()
gold = train[TARGETS].notna().all(axis=1).to_numpy()
fit_idx, gold_idx = np.flatnonzero(~gold), np.flatnonzero(gold)
if len(ids) != 4407 or len(set(ids)) != 4407 or len(fit_idx) != 4349 or len(gold_idx) != 58:
    raise RuntimeError("competition identity/count drift")

cache_ids = pd.read_csv(studies_path, dtype=str).StudyInstanceUID.astype(str).to_numpy()
if len(cache_ids) != 4407 or len(set(cache_ids)) != 4407 or set(cache_ids) != set(ids):
    raise RuntimeError("cache study manifest drift")
cache_row = {uid: i for i, uid in enumerate(cache_ids)}
features = []
with h5py.File(h5_path, "r") as handle:
    pixels, masks = handle["pixels"], handle["masks"]
    if pixels.shape != (4407, 6, 12, 288, 288) or masks.shape != (4407, 6):
        raise RuntimeError("physical cache shape drift")
    for output_row, uid in enumerate(ids):
        row = cache_row[uid]
        features.append(study_features(pixels[row], masks[row]))
        if output_row % 250 == 0:
            print(f"morphology features {output_row}/4407", flush=True)
features = np.asarray(features, np.float32)
if features.shape != (4407, 1356) or not np.isfinite(features).all():
    raise RuntimeError(f"feature contract drift: {features.shape}")

labels = pd.read_csv(unique_input("llm_labels_v4_blend.csv"),
                     dtype={"StudyInstanceUID": str})
labels = train[["StudyInstanceUID"]].merge(labels, on="StudyInstanceUID",
                                             how="left", validate="one_to_one")
steven = labels[TARGETS].to_numpy(np.float32)
if not np.isfinite(steven).all():
    raise RuntimeError("report labels incomplete")
oof = pd.read_csv(oof_path, dtype={"StudyInstanceUID": str})
if oof.columns.tolist() != ["StudyInstanceUID", "teacher_fold"] + TARGETS:
    raise RuntimeError("OOF schema drift")
if len(oof) != 2610 or oof.StudyInstanceUID.nunique() != 2610 or set(oof.teacher_fold) != {0, 1, 2}:
    raise RuntimeError("OOF coverage drift")
if set(oof.StudyInstanceUID) & set(ids[gold]):
    raise RuntimeError("gold leakage in OOF teacher")
teacher = train[["StudyInstanceUID"]].merge(oof, on="StudyInstanceUID",
                                             how="left", validate="one_to_one")
covered = teacher[TARGETS].notna().all(axis=1).to_numpy() & ~gold
if int(covered.sum()) != 2610:
    raise RuntimeError("OOF alignment drift")
corrected = steven.copy()
corrected[covered] = .5 * steven[covered] + .5 * teacher.loc[covered, TARGETS].to_numpy(np.float32)

anchor_pkg = np.load(anchor_path, allow_pickle=False)
gold_ids = anchor_pkg["ids"].astype(str)
arm_names = anchor_pkg["arm_names"].astype(str).tolist()
arm_pred = anchor_pkg["predictions"].astype(np.float64)
truth = anchor_pkg["truth"].astype(np.float64)
if arm_names != ANCHOR_ARMS or arm_pred.shape != (58, 4, 12):
    raise RuntimeError("anchor arm drift")
if gold_ids.tolist() != ids[gold].tolist() or not np.array_equal(truth, train.loc[gold, TARGETS].to_numpy(float)):
    raise RuntimeError("gold identity/truth drift")
if not np.isfinite(arm_pred).all() or arm_pred.min() < 0 or arm_pred.max() > 1:
    raise RuntimeError("anchor probability drift")
anchor_rank = ranks(np.einsum("nat,a->nt", arm_pred, ANCHOR_WEIGHTS))
anchor_auc = macro_auc(truth, anchor_rank)

seed_predictions, seed_rows = [], []
for seed in SEEDS:
    model = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=6, max_features=.50,
        bootstrap=True, max_samples=.85, n_jobs=-1, random_state=seed)
    model.fit(features[fit_idx], corrected[fit_idx])
    prediction = np.clip(model.predict(features[gold_idx]), 0, 1)
    blend = .90 * anchor_rank + .10 * ranks(prediction)
    row = {"seed": seed, "reader_gold_auc": macro_auc(truth, ranks(prediction)),
           "blend_gold_auc": macro_auc(truth, blend),
           "blend_minus_anchor": macro_auc(truth, blend) - anchor_auc}
    print(json.dumps(row), flush=True)
    seed_rows.append(row); seed_predictions.append(prediction)

reader = np.mean(seed_predictions, axis=0)
candidate = .90 * anchor_rank + .10 * ranks(reader)
candidate_auc = macro_auc(truth, candidate)
anchor_targets = target_aucs(truth, anchor_rank)
candidate_targets = target_aucs(truth, candidate)
delta = candidate_targets - anchor_targets
bootstrap = paired_bootstrap(truth, anchor_rank, candidate)
positive_seeds = int(sum(row["blend_minus_anchor"] > 0 for row in seed_rows))
gate = {
    "macro_gain_at_least_0_002": candidate_auc - anchor_auc >= .002,
    "nonnegative_targets_at_least_8": int((delta >= 0).sum()) >= 8,
    "positive_seeds_at_least_4": positive_seeds >= 4,
    "bootstrap_p_at_least_0_90": bootstrap["p_improve"] >= .90,
    "bootstrap_q025_above_minus_0_001": bootstrap["q025"] > -.001,
    "identity_integrity": True, "gold_fit_count_zero": True,
    "finite_predictions": bool(np.isfinite(candidate).all()),
}
report = {
    "status": "PASS" if all(gate.values()) else "REJECTED",
    "contract": "fixed quantitative morphology ExtraTrees reader; 10% rank residual; no submission",
    "h5_sha256": H5_SHA, "studies_sha256": STUDIES_SHA,
    "oof_sha256": OOF_SHA, "anchor_cache_sha256": ANCHOR_SHA,
    "feature_shape": list(features.shape), "fit_studies": 4349,
    "corrected_studies": 2610, "unchanged_studies": 1739,
    "gold_fit_count": 0, "gold_evaluation_count": 58,
    "anchor_arms": ANCHOR_ARMS, "anchor_weights": ANCHOR_WEIGHTS.tolist(),
    "residual_weight": RESIDUAL_WEIGHT, "seed_results": seed_rows,
    "ensemble": {"anchor_gold_auc": anchor_auc,
                 "reader_gold_auc": macro_auc(truth, ranks(reader)),
                 "candidate_gold_auc": candidate_auc,
                 "candidate_minus_anchor": candidate_auc - anchor_auc,
                 "positive_seeds": positive_seeds},
    "per_target": [{"target": TARGETS[j], "anchor_auc": float(anchor_targets[j]),
                    "candidate_auc": float(candidate_targets[j]), "delta": float(delta[j])}
                   for j in range(12)],
    "bootstrap": bootstrap, "gate": gate, "passed": bool(all(gate.values())),
    "submission_created": False,
}
np.savez_compressed(OUT / "quantitative_morphology_gold_predictions.npz",
                    ids=gold_ids, targets=np.asarray(TARGETS), truth=truth,
                    anchor=anchor_rank, reader=reader, candidate=candidate,
                    seeds=np.asarray(SEEDS), seed_predictions=np.asarray(seed_predictions))
(OUT / "quantitative_morphology_report.json").write_text(json.dumps(report, indent=2) + "\n")
for csv_path in OUT.glob("*.csv"):
    csv_path.unlink()
if list(OUT.glob("*.csv")):
    raise RuntimeError("forbidden CSV artifact")
print(json.dumps(report, indent=2), flush=True)
'''

source = FEATURE_SOURCE + "\n\n" + RUNNER
compile(source, "quantitative-morphology-specialist", "exec")
notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# Quantitative morphology specialist\n",
                    "Fixed five-seed physical-image screen; organizer gold is evaluation-only; no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run", "metadata": {},
         "outputs": [], "source": source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(ROOT / "quantitative-morphology-specialist.ipynb").write_text(json.dumps(notebook, indent=1))

metadata = {
    "id": "sarthaksharma14/rsna-knee-quantitative-morphology-specialist",
    "title": "RSNA Knee Quantitative Morphology Specialist",
    "code_file": "quantitative-morphology-specialist.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels"],
    "kernel_sources": ["sarthaksharma14/rsna-knee-288px-12-slice-cache",
                       "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit",
                       "sarthaksharma14/rsna-knee-multireader-nested-blend-screen"],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
