import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


source = r'''from pathlib import Path
import gc, hashlib, json, os, random, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
SEEDS = (20261031, 20261101, 20261102, 20261103, 20261104)
FEATURE_SHA = "7f1986bb30cf92dbbf440b43a5a5933c1a3d13a0bb61d91950457cb0f8e018cc"
CHECKPOINT_SHA = "0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80"
KNEE_SHA = "314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def unique_input(name):
    hits = list(Path("/kaggle/input").rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, found {len(hits)}: {hits[:8]}")
    return hits[0]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def macro_auc(y, p):
    scores = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) > 1:
            scores.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(scores))

def paired_bootstrap(y, control, candidate, draws=5000, seed=20261105):
    rng = np.random.default_rng(seed); delta = []
    for _ in range(draws):
        ix = rng.integers(0, len(y), len(y))
        delta.append(macro_auc(y[ix], candidate[ix]) - macro_auc(y[ix], control[ix]))
    delta = np.asarray(delta)
    return {"mean": float(delta.mean()),
            "p_improve": float((delta > 0).mean()),
            "q025": float(np.quantile(delta, .025)),
            "q975": float(np.quantile(delta, .975))}

if DEVICE.type != "cuda":
    raise RuntimeError("GPU required")
feature_path = unique_input("convnext_reader_fold0_features.npz")
if feature_path.stat().st_size != 363928367 or sha256(feature_path) != FEATURE_SHA:
    raise RuntimeError("feature artifact identity mismatch")
checkpoint_path = unique_input("cnxt_v0_fold0.pt")
if sha256(checkpoint_path) != CHECKPOINT_SHA:
    raise RuntimeError("checkpoint identity mismatch")
knee_path = unique_input("knee.py")
if sha256(knee_path) != KNEE_SHA:
    raise RuntimeError("model source identity mismatch")
sys.path.insert(0, str(knee_path.parent))
from knee import KneeNet

train_path = unique_input("train.csv")
train = pd.read_csv(train_path, dtype={"StudyInstanceUID": str})
ids = train.StudyInstanceUID.astype(str).to_numpy()
gold = train[TARGETS].notna().all(axis=1).to_numpy()
if len(ids) != 4407 or len(set(ids)) != 4407 or int(gold.sum()) != 58:
    raise RuntimeError("competition identity/gold drift")

package = np.load(feature_path, allow_pickle=False)
if package["ids"].astype(str).tolist() != ids.tolist():
    raise RuntimeError("feature study ordering mismatch")
features = package["features"]
masks = package["masks"]
positions = package["positions"]
if features.shape != (4407, 6, 12, 768) or features.dtype != np.float16:
    raise RuntimeError("feature shape/dtype drift")
if masks.shape != (4407, 6) or not np.isin(masks, [0, 1]).all():
    raise RuntimeError("mask drift")
if positions.shape != (4407, 6, 12) or not np.isfinite(positions).all():
    raise RuntimeError("position drift")

label_path = unique_input("llm_labels_v4_blend.csv")
label_frame = pd.read_csv(label_path, dtype={"StudyInstanceUID": str})
aligned = train[["StudyInstanceUID"]].merge(
    label_frame, on="StudyInstanceUID", how="left", validate="one_to_one")
steven = aligned[TARGETS].to_numpy(np.float32)
if not np.isfinite(steven).all():
    raise RuntimeError("Steven labels incomplete")

oof = pd.read_csv(unique_input("convnext_oof_predictions.csv"),
                  dtype={"StudyInstanceUID": str})
if oof.columns.tolist() != ["StudyInstanceUID", "teacher_fold"] + TARGETS:
    raise RuntimeError("OOF schema drift")
if len(oof) != 2610 or oof.StudyInstanceUID.nunique() != 2610:
    raise RuntimeError("OOF coverage drift")
if set(oof.teacher_fold.unique()) != {0, 1, 2}:
    raise RuntimeError("OOF fold drift")
if set(oof.StudyInstanceUID) & set(train.loc[gold, "StudyInstanceUID"]):
    raise RuntimeError("gold leakage in teacher table")
teacher = train[["StudyInstanceUID"]].merge(
    oof, on="StudyInstanceUID", how="left", validate="one_to_one")
covered = teacher[TARGETS].notna().all(axis=1).to_numpy()
covered_idx = np.flatnonzero(covered & ~gold)
fit_idx = np.flatnonzero(~gold)
if len(covered_idx) != 2610 or len(fit_idx) != 4349 or covered[gold].any():
    raise RuntimeError("teacher alignment failure")
corrected = steven.copy()
teacher_values = teacher[TARGETS].to_numpy(np.float32)
corrected[covered_idx] = .5 * steven[covered_idx] + .5 * teacher_values[covered_idx]

def new_head(seed):
    seed_all(seed)
    model = KneeNet("convnext_tiny", pretrained=False)
    del model.enc
    return model.to(DEVICE)

def head_forward(model, x, mask, pos):
    return model.head(x, mask.bool(), pos)

@torch.no_grad()
def predict(model, indices):
    model.eval(); output = []
    for start in range(0, len(indices), 128):
        ix = indices[start:start + 128]
        x = torch.from_numpy(features[ix]).to(DEVICE)
        m = torch.from_numpy(masks[ix]).to(DEVICE)
        p = torch.from_numpy(positions[ix]).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16):
            output.append(torch.sigmoid(head_forward(model, x, m, p)).float().cpu().numpy())
    return np.concatenate(output)

def train_fixed(labels, seed, epochs=20):
    model = new_head(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    dataset = TensorDataset(torch.from_numpy(features[fit_idx]),
                            torch.from_numpy(masks[fit_idx]),
                            torch.from_numpy(positions[fit_idx]),
                            torch.from_numpy(labels[fit_idx]))
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    losses = []
    for _ in range(epochs):
        model.train(); total = 0.0; seen = 0
        for x, mask, pos, y in loader:
            x, mask, pos, y = x.to(DEVICE), mask.to(DEVICE), pos.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = head_forward(model, x, mask, pos)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y.float())
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(x); seen += len(x)
        losses.append(total / seen)
    return model, losses

# Independently validate the original public fold-0 reader head on untouched gold.
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
original = KneeNet(checkpoint["args"]["backbone"].split(".")[0], pretrained=False)
original.load_state_dict(checkpoint["model"], strict=True)
del original.enc
original = original.to(DEVICE).eval()
gold_idx = np.flatnonzero(gold)
gold_truth = train.loc[gold, TARGETS].to_numpy(np.float64)
original_gold_prediction = predict(original, gold_idx)
original_gold_auc = macro_auc(gold_truth, original_gold_prediction)
del original; torch.cuda.empty_cache()

rows, control_gold, candidate_gold = [], [], []
for seed in SEEDS:
    control, control_loss = train_fixed(steven, seed)
    candidate, candidate_loss = train_fixed(corrected, seed)
    cp = predict(control, gold_idx); xp = predict(candidate, gold_idx)
    ca = macro_auc(gold_truth, cp); xa = macro_auc(gold_truth, xp)
    rows.append({"seed": seed, "fit_studies": 4349,
                 "corrected_studies": 2610, "unchanged_studies": 1739,
                 "epochs": 20, "control_final_loss": control_loss[-1],
                 "candidate_final_loss": candidate_loss[-1],
                 "control_gold_auc": ca, "candidate_gold_auc": xa,
                 "delta_candidate_minus_control": xa - ca})
    control_gold.append(cp); candidate_gold.append(xp)
    print(f"seed {seed}: corrected-control {xa-ca:+.6f}", flush=True)
    del control, candidate; torch.cuda.empty_cache(); gc.collect()

control_mean = np.mean(control_gold, axis=0)
candidate_mean = np.mean(candidate_gold, axis=0)
control_auc = macro_auc(gold_truth, control_mean)
candidate_auc = macro_auc(gold_truth, candidate_mean)
bootstrap = paired_bootstrap(gold_truth, control_mean, candidate_mean)
deltas = np.asarray([row["delta_candidate_minus_control"] for row in rows])
report = {
    "contract": "private paired ConvNeXt-reader head target audit; no submission",
    "feature_sha256": FEATURE_SHA,
    "checkpoint_sha256": CHECKPOINT_SHA,
    "source_knee_sha256": KNEE_SHA,
    "feature_shape": list(features.shape),
    "fit_studies_each_arm": 4349,
    "corrected_studies": 2610,
    "unchanged_studies": 1739,
    "gold_fit_count": 0,
    "gold_evaluation_count": 58,
    "checkpoint_selection": "none; fixed 20 epochs",
    "original_public_fold0_gold_auc": original_gold_auc,
    "seeds": rows,
    "ensemble": {"control_gold_auc": control_auc,
                 "candidate_gold_auc": candidate_auc,
                 "delta_candidate_minus_control": candidate_auc - control_auc,
                 "candidate_minus_original_public_fold0": candidate_auc - original_gold_auc},
    "bootstrap": bootstrap,
    "gate": {"delta_at_least_0_005": candidate_auc - control_auc >= .005,
             "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,
             "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= .95,
             "bootstrap_q025_positive": bootstrap["q025"] > 0,
             "candidate_absolute_at_least_0_90": candidate_auc >= .90},
    "submission_created": False,
}
report["passed"] = bool(all(report["gate"].values()))
Path("/kaggle/working/convnext_reader_oof_head_report.json").write_text(
    json.dumps(report, indent=2) + "\n")
if Path("/kaggle/working/submission.csv").exists():
    raise RuntimeError("forbidden submission artifact")
print(json.dumps(report, indent=2))
'''

notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# ConvNeXt reader OOF-corrected head validation\n",
                    "Private paired fixed-epoch audit; no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run",
         "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "convnext-reader-oof-head-validation.ipynb").write_text(json.dumps(notebook))

metadata = {
    "id": "sarthaksharma14/rsna-knee-convnext-reader-oof-head",
    "title": "RSNA Knee ConvNeXt Reader OOF Head",
    "code_file": "convnext-reader-oof-head-validation.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": "true",
    "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader",
                        "stevenleehans/rsna-knee-llm-report-labels"],
    "kernel_sources": ["sarthaksharma14/rsna-knee-convnext-reader-feature-cache",
                       "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit"],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

