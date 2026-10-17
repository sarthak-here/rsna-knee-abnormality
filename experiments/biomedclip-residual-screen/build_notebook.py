"""Build the frozen-BiomedCLIP target-attention residual screen."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = r'''from pathlib import Path
import gc, hashlib, json, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
SEEDS = (20261207, 20261208, 20261209, 20261210, 20261211)
EPOCHS = 30
FEATURE_SHA = "40768bf7c57478732c1fb14c77510b4e650634222337364c109747d0d3e9c434"
MODEL_SHA = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
OOF_SHA = "60529a0ba00107abd7efd101f70454e7ce89407befbaa4107f24771ee088579e"
ANCHOR_SHA = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"
ANCHOR_ARMS = ["r6_flip", "raptor384", "raptor224", "r7"]
ANCHOR_WEIGHTS = np.asarray([.50, .325, .175, 0.0], np.float64)
RESIDUAL_WEIGHT = .10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def paired_bootstrap(y, control, candidate, draws=10000, seed=20261212):
    rng = np.random.default_rng(seed); delta = []
    for _ in range(draws):
        ix = rng.integers(0, len(y), len(y)); scores = []
        for prediction in (control, candidate):
            valid = [roc_auc_score(y[ix, j], prediction[ix, j]) for j in range(12)
                     if len(np.unique(y[ix, j])) == 2]
            scores.append(np.mean(valid))
        delta.append(scores[1] - scores[0])
    delta = np.asarray(delta)
    return {"mean": float(delta.mean()), "p_improve": float((delta > 0).mean()),
            "q025": float(np.quantile(delta, .025)),
            "q975": float(np.quantile(delta, .975))}

class TargetAttention(nn.Module):
    def __init__(self, input_dim=512, dim=256, targets=12):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim),
                                     nn.GELU(), nn.Dropout(.10), nn.Linear(dim, dim))
        self.slot = nn.Parameter(torch.randn(6, dim) * .02)
        self.window = nn.Parameter(torch.randn(4, dim) * .02)
        self.query = nn.Parameter(torch.randn(targets, dim) * .02)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.output_weight = nn.Parameter(torch.randn(targets, dim) * .02)
        self.output_bias = nn.Parameter(torch.zeros(targets))

    def forward(self, features, masks):
        batch = features.shape[0]
        token = self.project(features.float())
        token = token + self.slot[None, :, None] + self.window[None, None]
        token = token.reshape(batch, 24, -1)
        valid = masks.bool().repeat_interleave(4, dim=1)
        score = torch.einsum("bnd,td->btn", self.key(token), self.query) / math.sqrt(token.shape[-1])
        score = score.masked_fill(~valid[:, None], -1e4)
        context = torch.einsum("btn,bnd->btd", torch.softmax(score, dim=-1), self.value(token))
        return (context * self.output_weight[None]).sum(-1) / math.sqrt(context.shape[-1]) + self.output_bias

if DEVICE.type != "cuda":
    raise RuntimeError("GPU required")
feature_path = unique_input("biomedclip_physical_features.npz")
oof_path = unique_input("convnext_oof_predictions.csv")
anchor_path = unique_input("multireader_gold_predictions.npz")
if sha256(feature_path) != FEATURE_SHA or sha256(oof_path) != OOF_SHA or sha256(anchor_path) != ANCHOR_SHA:
    raise RuntimeError("input identity drift")

train = pd.read_csv(unique_input("train.csv"), dtype={"StudyInstanceUID": str})
ids = train.StudyInstanceUID.astype(str).to_numpy()
gold = train[TARGETS].notna().all(axis=1).to_numpy()
fit_idx, gold_idx = np.flatnonzero(~gold), np.flatnonzero(gold)
if len(ids) != 4407 or len(set(ids)) != 4407 or len(fit_idx) != 4349 or len(gold_idx) != 58:
    raise RuntimeError("competition identity/count drift")

package = np.load(feature_path, allow_pickle=False)
cache_ids = package["ids"].astype(str)
raw_features, raw_masks = package["features"], package["masks"]
if cache_ids.dtype.kind not in ("U", "S") or len(set(cache_ids.tolist())) != 4407:
    raise RuntimeError("pickle-free cache ID drift")
if raw_features.shape != (4407, 6, 4, 512) or raw_features.dtype != np.float16:
    raise RuntimeError("feature tensor drift")
if raw_masks.shape != (4407, 6) or int(raw_masks.sum()) != 23089:
    raise RuntimeError("feature mask drift")
if str(package["model_sha256"]) != MODEL_SHA or not np.isfinite(raw_features).all():
    raise RuntimeError("model identity or feature finiteness drift")
row = {uid: index for index, uid in enumerate(cache_ids)}
if set(row) != set(ids):
    raise RuntimeError("feature study identity drift")
order = np.asarray([row[uid] for uid in ids])
features, masks = raw_features[order], raw_masks[order]

labels = pd.read_csv(unique_input("llm_labels_v4_blend.csv"), dtype={"StudyInstanceUID": str})
labels = train[["StudyInstanceUID"]].merge(labels, on="StudyInstanceUID", how="left", validate="one_to_one")
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
teacher = train[["StudyInstanceUID"]].merge(oof, on="StudyInstanceUID", how="left", validate="one_to_one")
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

dataset = TensorDataset(torch.from_numpy(features[fit_idx]), torch.from_numpy(masks[fit_idx]),
                        torch.from_numpy(corrected[fit_idx]))

@torch.inference_mode()
def predict(model, indices):
    model.eval(); output = []
    for start in range(0, len(indices), 256):
        ix = indices[start:start + 256]
        x = torch.from_numpy(features[ix]).to(DEVICE)
        m = torch.from_numpy(masks[ix]).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16):
            output.append(torch.sigmoid(model(x, m)).float().cpu().numpy())
    return np.concatenate(output)

def train_seed(seed):
    seed_all(seed)
    model = TargetAttention().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-3)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed), pin_memory=True)
    scaler = torch.amp.GradScaler("cuda"); history = []
    for epoch in range(EPOCHS):
        model.train(); total = 0.; seen = 0
        for x, mask, target in loader:
            x = x.to(DEVICE, non_blocking=True); mask = mask.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, mask)
                loss = F.binary_cross_entropy_with_logits(logits.float(), target.float())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(x); seen += len(x)
        history.append(total / seen)
        print(f"seed={seed} epoch={epoch+1}/{EPOCHS} loss={history[-1]:.6f}", flush=True)
    prediction = predict(model, gold_idx)
    checkpoint = OUT / f"biomedclip_head_seed_{seed}.pt"
    torch.save({"seed": seed, "epochs": EPOCHS,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, checkpoint)
    return prediction, history, sha256(checkpoint)

seed_predictions, seed_rows = [], []
for seed in SEEDS:
    prediction, history, checkpoint_sha = train_seed(seed)
    blend = .90 * anchor_rank + .10 * ranks(prediction)
    row = {"seed": seed, "epochs": EPOCHS, "final_loss": history[-1],
           "reader_gold_auc": macro_auc(truth, ranks(prediction)),
           "blend_gold_auc": macro_auc(truth, blend),
           "blend_minus_anchor": macro_auc(truth, blend) - anchor_auc,
           "checkpoint_sha256": checkpoint_sha}
    print(json.dumps(row), flush=True)
    seed_rows.append(row); seed_predictions.append(prediction)
    gc.collect(); torch.cuda.empty_cache()

reader = np.mean(seed_predictions, axis=0)
candidate = .90 * anchor_rank + .10 * ranks(reader)
candidate_auc = macro_auc(truth, candidate)
anchor_targets, candidate_targets = target_aucs(truth, anchor_rank), target_aucs(truth, candidate)
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
    "contract": "frozen BiomedCLIP target-attention reader; fixed 10% rank residual; no submission",
    "feature_sha256": FEATURE_SHA, "model_sha256": MODEL_SHA,
    "oof_sha256": OOF_SHA, "anchor_cache_sha256": ANCHOR_SHA,
    "fit_studies": 4349, "corrected_studies": 2610, "unchanged_studies": 1739,
    "gold_fit_count": 0, "gold_evaluation_count": 58,
    "feature_shape": list(features.shape), "anchor_arms": ANCHOR_ARMS,
    "anchor_weights": ANCHOR_WEIGHTS.tolist(), "residual_weight": RESIDUAL_WEIGHT,
    "fixed_epochs": EPOCHS, "seed_results": seed_rows,
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
np.savez_compressed(OUT / "biomedclip_residual_gold_predictions.npz",
                    ids=gold_ids.astype("U"), targets=np.asarray(TARGETS, dtype="U"), truth=truth,
                    anchor=anchor_rank, reader=reader, candidate=candidate,
                    seeds=np.asarray(SEEDS), seed_predictions=np.asarray(seed_predictions))
(OUT / "biomedclip_residual_report.json").write_text(json.dumps(report, indent=2) + "\n")
for csv_path in OUT.glob("*.csv"):
    csv_path.unlink()
if list(OUT.glob("*.csv")):
    raise RuntimeError("forbidden CSV artifact")
print(json.dumps(report, indent=2), flush=True)
'''

compile(SOURCE, "biomedclip-residual-screen", "exec")
notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# BiomedCLIP target-attention residual screen\n",
                    "Fixed five-seed validation; organizer gold is evaluation-only; no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run", "metadata": {},
         "outputs": [], "source": SOURCE.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(ROOT / "biomedclip-residual-screen.ipynb").write_text(json.dumps(notebook, indent=1))
metadata = {
    "id": "sarthaksharma14/rsna-knee-biomedclip-residual-screen",
    "title": "RSNA Knee BiomedCLIP Residual Screen",
    "code_file": "biomedclip-residual-screen.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels"],
    "kernel_sources": ["sarthaksharma14/rsna-knee-biomedclip-feature-cache",
                       "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit",
                       "sarthaksharma14/rsna-knee-multireader-nested-blend-screen"],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
