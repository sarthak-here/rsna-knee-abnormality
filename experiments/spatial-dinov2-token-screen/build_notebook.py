"""Build the frozen spatial-patch DINOv2 residual screen notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = r'''from pathlib import Path
import gc, hashlib, json, math, random, time
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
SEEDS = (20261221, 20261222, 20261223, 20261224, 20261225)
EPOCHS = 20
H5_SHA = "490e5ccc07143699d05a9eb3ef9583e08795f1c2bacc6d9cf2bd0a9268158526"
STUDIES_SHA = "efc63afee30345f94a916f98ba6b30f266fccfe2976b8475fa86653b2c510513"
OOF_SHA = "60529a0ba00107abd7efd101f70454e7ce89407befbaa4107f24771ee088579e"
ANCHOR_SHA = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"
ANCHOR_ARMS = ["r6_flip", "raptor384", "raptor224", "r7"]
ANCHOR_WEIGHTS = np.asarray([.50, .325, .175, 0.0], np.float64)
RESIDUAL_WEIGHT = .10
FEATURE_SHAPE = (4407, 6, 4, 16, 384)
OCCUPIED_SLOTS = 23089
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working")
START = time.time()

def log(message):
    print(f"[{time.time() - START:8.1f}s] {message}", flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
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

def paired_bootstrap(y, control, candidate, draws=10000, seed=20261226):
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

def model_root():
    roots = sorted(set(path.parent for path in Path("/kaggle/input").rglob("config.json")
                       if "dinov2" in str(path).lower()))
    if len(roots) != 1:
        raise RuntimeError(f"expected one DINOv2 model source, found {roots}")
    return roots[0]

def asset_hashes(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                         "sha256": sha256(path)})
    if not rows:
        raise RuntimeError("DINOv2 model source is empty")
    return rows

@torch.inference_mode()
def build_spatial_features(h5_path, encoder):
    path = OUT / "spatial_dinov2_features.npy"
    feature = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                        shape=FEATURE_SHAPE)
    feature[:] = 0
    mean = torch.tensor([.485, .456, .406], device=DEVICE)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=DEVICE)[None, :, None, None]
    with h5py.File(h5_path, "r", swmr=True) as handle:
        pixels, masks = handle["pixels"], handle["masks"]
        if pixels.shape != (4407, 6, 12, 288, 288) or masks.shape != (4407, 6):
            raise RuntimeError("physical cache shape drift")
        if int(np.asarray(masks).sum()) != OCCUPIED_SLOTS:
            raise RuntimeError("physical cache occupancy drift")
        for first in range(0, 4407, 4):
            stop = min(first + 4, 4407)
            study = torch.from_numpy(np.asarray(pixels[first:stop]))
            mask = torch.from_numpy(np.asarray(masks[first:stop])).bool()
            images = study.reshape(stop - first, 6, 4, 3, 288, 288)
            valid = mask[:, :, None].expand(-1, -1, 4)
            locations = valid.nonzero(as_tuple=False)
            images = images[valid]
            values = []
            for offset in range(0, len(images), 24):
                x = images[offset:offset + 24].to(DEVICE).float().div_(255.)
                x = (x - mean) / std
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden = encoder(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
                    patch = hidden[:, 1:]
                    side = int(round(math.sqrt(patch.shape[1])))
                    if side * side != patch.shape[1] or side != 20 or patch.shape[2] != 384:
                        raise RuntimeError(f"DINOv2 patch contract drift: {tuple(patch.shape)}")
                    patch = patch.transpose(1, 2).reshape(len(patch), 384, side, side)
                    pooled = F.adaptive_avg_pool2d(patch, (4, 4)).flatten(2).transpose(1, 2)
                values.append(pooled.float().cpu())
            values = torch.cat(values).numpy().astype(np.float16)
            for location, value in zip(locations.numpy(), values):
                b, slot, window = location
                feature[first + b, slot, window] = value
            if stop % 100 <= 4 or stop == 4407:
                feature.flush(); log(f"encoded {stop}/4407 studies")
    feature.flush(); del feature; gc.collect()
    return path

class SpatialRows(Dataset):
    def __init__(self, features, masks, labels, indices):
        self.features, self.masks, self.labels = features, masks, labels
        self.indices = np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, item):
        row = self.indices[item]
        return (torch.from_numpy(np.array(self.features[row], copy=True)),
                torch.from_numpy(np.array(self.masks[row], copy=True)),
                torch.from_numpy(self.labels[row]))

class SpatialTargetHead(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, dim),
                                     nn.GELU(), nn.Dropout(.10), nn.Linear(dim, dim))
        self.slot = nn.Parameter(torch.randn(6, dim) * .02)
        self.window = nn.Parameter(torch.randn(4, dim) * .02)
        self.row = nn.Parameter(torch.randn(4, dim) * .02)
        self.col = nn.Parameter(torch.randn(4, dim) * .02)
        self.query = nn.Parameter(torch.randn(12, dim) * .02)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.output_weight = nn.Parameter(torch.randn(12, dim) * .02)
        self.output_bias = nn.Parameter(torch.zeros(12))
    def forward(self, feature, mask):
        token = self.project(feature.float())
        grid = (self.row[:, None] + self.col[None, :]).reshape(16, -1)
        token = token + self.slot[None, :, None, None] + self.window[None, None, :, None] + grid[None, None, None]
        token = token.reshape(len(token), 384, -1)
        valid = mask.bool()[:, :, None, None].expand(-1, -1, 4, 16).reshape(len(mask), 384)
        score = torch.einsum("bnd,td->btn", self.key(token), self.query) / math.sqrt(token.shape[-1])
        score = score.masked_fill(~valid[:, None], -1e4)
        context = torch.einsum("btn,bnd->btd", torch.softmax(score, -1), self.value(token))
        return (context * self.output_weight[None]).sum(-1) / math.sqrt(context.shape[-1]) + self.output_bias

if DEVICE.type != "cuda":
    raise RuntimeError("GPU required")
h5_path = unique_input("train_288x12.h5")
studies_path = unique_input("highres_cache_studies.csv")
oof_path = unique_input("convnext_oof_predictions.csv")
anchor_path = unique_input("multireader_gold_predictions.npz")
if h5_path.stat().st_size != 17051786047 or sha256(h5_path) != H5_SHA:
    raise RuntimeError("physical cache identity drift")
if sha256(studies_path) != STUDIES_SHA or sha256(oof_path) != OOF_SHA or sha256(anchor_path) != ANCHOR_SHA:
    raise RuntimeError("input identity drift")

train = pd.read_csv(unique_input("train.csv"), dtype={"StudyInstanceUID": str})
ids = train.StudyInstanceUID.astype(str).to_numpy()
cache_ids = pd.read_csv(studies_path, dtype=str).StudyInstanceUID.astype(str).to_numpy()
if ids.tolist() != cache_ids.tolist() or len(ids) != 4407 or len(set(ids)) != 4407:
    raise RuntimeError("study identity/order drift")
gold = train[TARGETS].notna().all(axis=1).to_numpy()
fit_idx, gold_idx = np.flatnonzero(~gold), np.flatnonzero(gold)
if len(fit_idx) != 4349 or len(gold_idx) != 58:
    raise RuntimeError("gold/non-gold count drift")

root = model_root(); assets = asset_hashes(root)
encoder = AutoModel.from_pretrained(str(root), local_files_only=True).to(DEVICE).eval()
for parameter in encoder.parameters(): parameter.requires_grad_(False)
if int(encoder.config.hidden_size) != 384:
    raise RuntimeError("DINOv2-S hidden size drift")
feature_path = build_spatial_features(h5_path, encoder)
feature_sha = sha256(feature_path)
del encoder; gc.collect(); torch.cuda.empty_cache()
features = np.load(feature_path, mmap_mode="r")
with h5py.File(h5_path, "r", swmr=True) as handle:
    masks = np.asarray(handle["masks"], dtype=bool)
if features.shape != FEATURE_SHAPE or features.dtype != np.float16 or not np.isfinite(features).all():
    raise RuntimeError("spatial feature cache integrity failure")

labels = pd.read_csv(unique_input("llm_labels_v4_blend.csv"), dtype={"StudyInstanceUID": str})
labels = train[["StudyInstanceUID"]].merge(labels, on="StudyInstanceUID", how="left", validate="one_to_one")
steven = labels[TARGETS].to_numpy(np.float32)
if not np.isfinite(steven).all(): raise RuntimeError("report labels incomplete")
oof = pd.read_csv(oof_path, dtype={"StudyInstanceUID": str})
if oof.columns.tolist() != ["StudyInstanceUID", "teacher_fold"] + TARGETS:
    raise RuntimeError("OOF schema drift")
if len(oof) != 2610 or oof.StudyInstanceUID.nunique() != 2610 or set(oof.teacher_fold) != {0, 1, 2}:
    raise RuntimeError("OOF coverage drift")
if set(oof.StudyInstanceUID) & set(ids[gold]): raise RuntimeError("gold leakage in OOF teacher")
teacher = train[["StudyInstanceUID"]].merge(oof, on="StudyInstanceUID", how="left", validate="one_to_one")
covered = teacher[TARGETS].notna().all(axis=1).to_numpy() & ~gold
if int(covered.sum()) != 2610: raise RuntimeError("OOF alignment drift")
corrected = steven.copy()
corrected[covered] = .5 * steven[covered] + .5 * teacher.loc[covered, TARGETS].to_numpy(np.float32)

anchor_pkg = np.load(anchor_path, allow_pickle=False)
gold_ids = anchor_pkg["ids"].astype(str)
arm_names = anchor_pkg["arm_names"].astype(str).tolist()
arm_pred = anchor_pkg["predictions"].astype(np.float64)
truth = anchor_pkg["truth"].astype(np.float64)
if arm_names != ANCHOR_ARMS or arm_pred.shape != (58, 4, 12): raise RuntimeError("anchor arm drift")
if gold_ids.tolist() != ids[gold].tolist() or not np.array_equal(truth, train.loc[gold, TARGETS].to_numpy(float)):
    raise RuntimeError("gold identity/truth drift")
anchor_rank = ranks(np.einsum("nat,a->nt", arm_pred, ANCHOR_WEIGHTS))
anchor_auc = macro_auc(truth, anchor_rank)

@torch.inference_mode()
def predict(model, indices):
    model.eval(); output = []
    data = SpatialRows(features, masks, corrected, indices)
    for x, mask, _ in DataLoader(data, batch_size=64, shuffle=False, num_workers=0):
        with torch.autocast("cuda", dtype=torch.float16):
            output.append(torch.sigmoid(model(x.to(DEVICE), mask.to(DEVICE))).float().cpu().numpy())
    return np.concatenate(output)

def train_seed(seed):
    seed_all(seed); model = SpatialTargetHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-3)
    loader = DataLoader(SpatialRows(features, masks, corrected, fit_idx), batch_size=32,
                        shuffle=True, num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(seed))
    scaler = torch.amp.GradScaler("cuda"); history = []
    for epoch in range(EPOCHS):
        model.train(); total = 0.; seen = 0
        for x, mask, target in loader:
            x=x.to(DEVICE, non_blocking=True); mask=mask.to(DEVICE, non_blocking=True); target=target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits=model(x, mask); loss=F.binary_cross_entropy_with_logits(logits.float(), target.float())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(x); seen += len(x)
        history.append(total / seen); log(f"seed={seed} epoch={epoch+1}/{EPOCHS} loss={history[-1]:.6f}")
    prediction = predict(model, gold_idx)
    checkpoint = OUT / f"spatial_dinov2_head_seed_{seed}.pt"
    torch.save({"seed": seed, "epochs": EPOCHS,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, checkpoint)
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
    print(json.dumps(row), flush=True); seed_rows.append(row); seed_predictions.append(prediction)
    gc.collect(); torch.cuda.empty_cache()

reader = np.mean(seed_predictions, axis=0)
candidate = .90 * anchor_rank + .10 * ranks(reader)
candidate_auc = macro_auc(truth, candidate)
anchor_targets, candidate_targets = target_aucs(truth, anchor_rank), target_aucs(truth, candidate)
delta = candidate_targets - anchor_targets
bootstrap = paired_bootstrap(truth, anchor_rank, candidate)
positive_seeds = int(sum(row["blend_minus_anchor"] > 0 for row in seed_rows))
gate = {"macro_gain_at_least_0_002": candidate_auc-anchor_auc >= .002,
        "nonnegative_targets_at_least_8": int((delta >= 0).sum()) >= 8,
        "positive_seeds_at_least_4": positive_seeds >= 4,
        "bootstrap_p_at_least_0_90": bootstrap["p_improve"] >= .90,
        "bootstrap_q025_above_minus_0_001": bootstrap["q025"] > -.001,
        "identity_integrity": True, "gold_fit_count_zero": True,
        "finite_predictions": bool(np.isfinite(candidate).all())}
report = {"status": "PASS" if all(gate.values()) else "REJECTED",
          "contract": "frozen DINOv2-S 4x4 spatial patch-grid target attention; fixed 10% rank residual; no submission",
          "h5_sha256": H5_SHA, "studies_sha256": STUDIES_SHA,
          "feature_sha256": feature_sha, "feature_shape": list(features.shape),
          "dinov2_model_source": "metaresearch/dinov2/PyTorch/small/1",
          "dinov2_assets": assets, "oof_sha256": OOF_SHA, "anchor_cache_sha256": ANCHOR_SHA,
          "fit_studies": 4349, "corrected_studies": 2610, "unchanged_studies": 1739,
          "gold_fit_count": 0, "gold_evaluation_count": 58, "occupied_slots": OCCUPIED_SLOTS,
          "anchor_arms": ANCHOR_ARMS, "anchor_weights": ANCHOR_WEIGHTS.tolist(),
          "residual_weight": RESIDUAL_WEIGHT, "fixed_epochs": EPOCHS, "seed_results": seed_rows,
          "ensemble": {"anchor_gold_auc": anchor_auc, "reader_gold_auc": macro_auc(truth, ranks(reader)),
                       "candidate_gold_auc": candidate_auc, "candidate_minus_anchor": candidate_auc-anchor_auc,
                       "positive_seeds": positive_seeds},
          "per_target": [{"target": TARGETS[j], "anchor_auc": float(anchor_targets[j]),
                          "candidate_auc": float(candidate_targets[j]), "delta": float(delta[j])}
                         for j in range(12)],
          "bootstrap": bootstrap, "gate": gate, "passed": bool(all(gate.values())),
          "submission_created": False}
np.savez_compressed(OUT / "spatial_dinov2_gold_predictions.npz",
                    ids=gold_ids.astype("U"), targets=np.asarray(TARGETS, dtype="U"), truth=truth,
                    anchor=anchor_rank, reader=reader, candidate=candidate,
                    seeds=np.asarray(SEEDS), seed_predictions=np.asarray(seed_predictions))
(OUT / "spatial_dinov2_report.json").write_text(json.dumps(report, indent=2) + "\n")
for csv_path in OUT.glob("*.csv"): csv_path.unlink()
if list(OUT.glob("*.csv")): raise RuntimeError("forbidden CSV artifact")
print(json.dumps(report, indent=2), flush=True)
'''

compile(SOURCE, "spatial-dinov2-token-screen", "exec")
notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# Spatial DINOv2 token residual screen\n",
                    "Frozen 4x4 patch-grid representation; organizer gold is evaluation-only; no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run", "metadata": {},
         "outputs": [], "source": SOURCE.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(ROOT / "spatial-dinov2-token-screen.ipynb").write_text(json.dumps(notebook, indent=1))
metadata = {
    "id": "sarthaksharma14/rsna-knee-spatial-dinov2-token-screen",
    "title": "RSNA Knee Spatial DINOv2 Token Screen",
    "code_file": "spatial-dinov2-token-screen.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels"],
    "kernel_sources": ["sarthaksharma14/rsna-knee-288px-12-slice-cache",
                       "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit",
                       "sarthaksharma14/rsna-knee-multireader-nested-blend-screen"],
    "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

