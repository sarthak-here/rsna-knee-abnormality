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
SEEDS = (20261117, 20261118, 20261119, 20261120, 20261121)
EPOCHS = 24
FEATURE_SHA = "7f1986bb30cf92dbbf440b43a5a5933c1a3d13a0bb61d91950457cb0f8e018cc"
OOF_SHA = "60529a0ba00107abd7efd101f70454e7ce89407befbaa4107f24771ee088579e"
ANCHOR_SHA = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"
ANCHOR_ARMS = ["r6_flip", "raptor384", "raptor224", "r7"]
ANCHOR_WEIGHTS = np.asarray([.50, .325, .175, 0.0], np.float64)
RESIDUAL_WEIGHT = .10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def unique_input(name):
    hits = [p for p in Path("/kaggle/input").rglob(name) if p.is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(hits)}: {hits[:8]}")
    return hits[0]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j])
                          for j in range(y.shape[1])]))

def target_aucs(y, p):
    return np.asarray([roc_auc_score(y[:, j], p[:, j])
                       for j in range(y.shape[1])], np.float64)

def ranks(p):
    out = np.empty_like(p, dtype=np.float64)
    for j in range(p.shape[1]):
        order = np.argsort(p[:, j], kind="mergesort")
        sorted_values = p[order, j]
        ranked = np.empty(len(p), np.float64)
        start = 0
        while start < len(p):
            stop = start + 1
            while stop < len(p) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranked[order[start:stop]] = .5 * (start + stop - 1)
            start = stop
        out[:, j] = (ranked + .5) / len(p)
    return out

def paired_bootstrap(y, control, candidate, draws=10000, seed=20261122):
    rng = np.random.default_rng(seed); delta = []
    for _ in range(draws):
        ix = rng.integers(0, len(y), len(y))
        scores = []
        for pred in (control, candidate):
            per_target = []
            for j in range(y.shape[1]):
                if len(np.unique(y[ix, j])) == 2:
                    per_target.append(roc_auc_score(y[ix, j], pred[ix, j]))
            scores.append(np.mean(per_target))
        delta.append(scores[1] - scores[0])
    delta = np.asarray(delta)
    return {"mean": float(delta.mean()),
            "p_improve": float((delta > 0).mean()),
            "q025": float(np.quantile(delta, .025)),
            "q975": float(np.quantile(delta, .975))}

class CrossPlaneTargetQuery(nn.Module):
    def __init__(self, in_dim=768, dim=192, targets=12):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.slot = nn.Parameter(torch.randn(6, dim) * .02)
        self.pos = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        intra_layer = nn.TransformerEncoderLayer(
            dim, 6, dim_feedforward=dim * 3, dropout=.10,
            activation="gelu", batch_first=True, norm_first=True)
        cross_layer = nn.TransformerEncoderLayer(
            dim, 6, dim_feedforward=dim * 3, dropout=.10,
            activation="gelu", batch_first=True, norm_first=True)
        self.intra = nn.TransformerEncoder(intra_layer, 2, enable_nested_tensor=False)
        self.cross = nn.TransformerEncoder(cross_layer, 2, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(targets, dim) * .02)
        self.query_attn = nn.MultiheadAttention(dim, 6, dropout=.10, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.out_weight = nn.Parameter(torch.randn(targets, dim) * .02)
        self.out_bias = nn.Parameter(torch.zeros(targets))
        self.dropout = nn.Dropout(.10)

    def forward(self, x, mask, positions):
        b = x.shape[0]
        z = self.proj(x.float())
        z = z + self.slot[None, :, None, :]
        z = z + self.pos(positions.float().unsqueeze(-1))
        z = self.intra(z.reshape(b * 6, 12, -1)).reshape(b, 6, 12, -1)
        z = z * mask[:, :, None, None].float()
        summaries = z.mean(2)
        summaries = self.cross(summaries, src_key_padding_mask=~mask.bool())
        tokens = (z + summaries[:, :, None, :]).reshape(b, 72, -1)
        valid = mask.bool().repeat_interleave(12, dim=1)
        q = self.queries[None].expand(b, -1, -1)
        context, _ = self.query_attn(q, tokens, tokens,
                                     key_padding_mask=~valid, need_weights=False)
        context = self.query_norm(context + q)
        return (self.dropout(context) * self.out_weight[None]).sum(-1) / math.sqrt(context.shape[-1]) + self.out_bias

if DEVICE.type != "cuda":
    raise RuntimeError("GPU required")

feature_path = unique_input("convnext_reader_fold0_features.npz")
oof_path = unique_input("convnext_oof_predictions.csv")
anchor_path = unique_input("multireader_gold_predictions.npz")
if sha256(feature_path) != FEATURE_SHA:
    raise RuntimeError("feature cache SHA drift")
if sha256(oof_path) != OOF_SHA:
    raise RuntimeError("OOF teacher SHA drift")
if sha256(anchor_path) != ANCHOR_SHA:
    raise RuntimeError("anchor cache SHA drift")

train_path = unique_input("train.csv")
train = pd.read_csv(train_path, dtype={"StudyInstanceUID": str})
ids = train.StudyInstanceUID.astype(str).to_numpy()
gold = train[TARGETS].notna().all(axis=1).to_numpy()
fit_idx = np.flatnonzero(~gold); gold_idx = np.flatnonzero(gold)
if len(ids) != 4407 or len(set(ids)) != 4407 or len(fit_idx) != 4349 or len(gold_idx) != 58:
    raise RuntimeError("competition identity/count drift")

package = np.load(feature_path, allow_pickle=False)
features = package["features"]
masks = package["masks"]
positions = package["positions"]
if package["ids"].astype(str).tolist() != ids.tolist():
    raise RuntimeError("feature ordering drift")
if features.shape != (4407, 6, 12, 768) or features.dtype != np.float16:
    raise RuntimeError("feature tensor drift")
if masks.shape != (4407, 6) or not np.isin(masks, [0, 1]).all() or (masks.sum(1) == 0).any():
    raise RuntimeError("slot mask drift")
if positions.shape != (4407, 6, 12) or not np.isfinite(positions).all():
    raise RuntimeError("position tensor drift")

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
    raise RuntimeError("organizer-gold leakage in OOF teacher")
teacher = train[["StudyInstanceUID"]].merge(oof, on="StudyInstanceUID",
                                             how="left", validate="one_to_one")
covered = teacher[TARGETS].notna().all(axis=1).to_numpy() & ~gold
if int(covered.sum()) != 2610:
    raise RuntimeError("OOF alignment drift")
corrected = steven.copy()
corrected[covered] = .5 * steven[covered] + .5 * teacher.loc[covered, TARGETS].to_numpy(np.float32)

anchor_pkg = np.load(anchor_path, allow_pickle=False)
required = {"ids", "targets", "truth", "arm_names", "predictions"}
if not required.issubset(set(anchor_pkg.files)):
    raise RuntimeError(f"anchor keys drift: {anchor_pkg.files}")
gold_ids = anchor_pkg["ids"].astype(str)
arm_names = anchor_pkg["arm_names"].astype(str).tolist()
arm_pred = anchor_pkg["predictions"].astype(np.float64)
truth = anchor_pkg["truth"].astype(np.float64)
if arm_names != ANCHOR_ARMS or arm_pred.shape != (58, 4, 12):
    raise RuntimeError("anchor arm contract drift")
if gold_ids.tolist() != ids[gold].tolist() or not np.array_equal(truth, train.loc[gold, TARGETS].to_numpy(np.float64)):
    raise RuntimeError("anchor gold identity/truth drift")
if not np.isfinite(arm_pred).all() or arm_pred.min() < 0 or arm_pred.max() > 1:
    raise RuntimeError("anchor probability drift")
anchor_raw = np.einsum("nat,a->nt", arm_pred, ANCHOR_WEIGHTS)
anchor_rank = ranks(anchor_raw)
anchor_auc = macro_auc(truth, anchor_rank)

dataset = TensorDataset(torch.from_numpy(features[fit_idx]),
                        torch.from_numpy(masks[fit_idx]),
                        torch.from_numpy(positions[fit_idx]),
                        torch.from_numpy(corrected[fit_idx]))

@torch.no_grad()
def predict(model, indices):
    model.eval(); output = []
    for start in range(0, len(indices), 64):
        ix = indices[start:start + 64]
        x = torch.from_numpy(features[ix]).to(DEVICE)
        m = torch.from_numpy(masks[ix]).to(DEVICE)
        p = torch.from_numpy(positions[ix]).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16):
            output.append(torch.sigmoid(model(x, m, p)).float().cpu().numpy())
    return np.concatenate(output)

def train_seed(seed):
    seed_all(seed)
    model = CrossPlaneTargetQuery().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-3)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed), pin_memory=True)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    for epoch in range(EPOCHS):
        model.train(); total = 0.; seen = 0
        for x, mask, pos, y in loader:
            x = x.to(DEVICE, non_blocking=True); mask = mask.to(DEVICE, non_blocking=True)
            pos = pos.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, mask, pos)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(x); seen += len(x)
        history.append(total / seen)
        print(f"seed={seed} epoch={epoch+1}/{EPOCHS} loss={history[-1]:.6f}", flush=True)
    pred = predict(model, gold_idx)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt = OUT / f"crossplane_seed_{seed}.pt"
    torch.save({"seed": seed, "epochs": EPOCHS, "model": state}, ckpt)
    return pred, history, sha256(ckpt)

rows = []; reader_predictions = []; seed_blends = []
for seed in SEEDS:
    pred, history, checkpoint_sha = train_seed(seed)
    pred_rank = ranks(pred)
    blend = (1. - RESIDUAL_WEIGHT) * anchor_rank + RESIDUAL_WEIGHT * pred_rank
    reader_auc = macro_auc(truth, pred_rank)
    blend_auc = macro_auc(truth, blend)
    rows.append({"seed": seed, "epochs": EPOCHS,
                 "final_loss": history[-1], "reader_gold_auc": reader_auc,
                 "blend_gold_auc": blend_auc,
                 "blend_minus_anchor": blend_auc - anchor_auc,
                 "checkpoint_sha256": checkpoint_sha})
    reader_predictions.append(pred); seed_blends.append(blend)
    print(f"seed={seed} reader={reader_auc:.6f} blend_delta={blend_auc-anchor_auc:+.6f}", flush=True)
    gc.collect(); torch.cuda.empty_cache()

reader_mean = np.mean(reader_predictions, axis=0)
candidate = (1. - RESIDUAL_WEIGHT) * anchor_rank + RESIDUAL_WEIGHT * ranks(reader_mean)
candidate_auc = macro_auc(truth, candidate)
anchor_targets = target_aucs(truth, anchor_rank)
candidate_targets = target_aucs(truth, candidate)
target_delta = candidate_targets - anchor_targets
bootstrap = paired_bootstrap(truth, anchor_rank, candidate)
seed_positive = int(sum(row["blend_minus_anchor"] > 0 for row in rows))
gate = {
    "macro_gain_at_least_0_002": candidate_auc - anchor_auc >= .002,
    "nonnegative_targets_at_least_8": int((target_delta >= 0).sum()) >= 8,
    "positive_seeds_at_least_4": seed_positive >= 4,
    "bootstrap_p_at_least_0_90": bootstrap["p_improve"] >= .90,
    "bootstrap_q025_above_minus_0_001": bootstrap["q025"] > -.001,
    "identity_integrity": True,
    "gold_fit_count_zero": True,
    "finite_predictions": bool(np.isfinite(candidate).all()),
}
report = {
    "status": "PASS" if all(gate.values()) else "REJECTED",
    "contract": "fixed cross-plane target-query reader; 10% rank residual; no submission",
    "feature_sha256": FEATURE_SHA,
    "oof_sha256": OOF_SHA,
    "anchor_cache_sha256": ANCHOR_SHA,
    "fit_studies": 4349,
    "corrected_studies": 2610,
    "unchanged_studies": 1739,
    "gold_fit_count": 0,
    "gold_evaluation_count": 58,
    "feature_shape": list(features.shape),
    "anchor_arms": ANCHOR_ARMS,
    "anchor_weights": ANCHOR_WEIGHTS.tolist(),
    "residual_weight": RESIDUAL_WEIGHT,
    "fixed_epochs": EPOCHS,
    "seed_results": rows,
    "ensemble": {"anchor_gold_auc": anchor_auc,
                 "reader_gold_auc": macro_auc(truth, ranks(reader_mean)),
                 "candidate_gold_auc": candidate_auc,
                 "candidate_minus_anchor": candidate_auc - anchor_auc,
                 "positive_seeds": seed_positive},
    "per_target": [{"target": TARGETS[j],
                    "anchor_auc": float(anchor_targets[j]),
                    "candidate_auc": float(candidate_targets[j]),
                    "delta": float(target_delta[j])} for j in range(12)],
    "bootstrap": bootstrap,
    "gate": gate,
    "passed": bool(all(gate.values())),
    "submission_created": False,
}
np.savez_compressed(OUT / "crossplane_gold_predictions.npz",
                    ids=gold_ids, targets=np.asarray(TARGETS), truth=truth,
                    anchor=anchor_rank, reader=reader_mean, candidate=candidate,
                    seeds=np.asarray(SEEDS), seed_predictions=np.asarray(reader_predictions))
(OUT / "crossplane_token_residual_report.json").write_text(json.dumps(report, indent=2) + "\n")
for csv_path in OUT.glob("*.csv"):
    csv_path.unlink()
if list(OUT.glob("*.csv")):
    raise RuntimeError("forbidden CSV artifact")
print(json.dumps(report, indent=2), flush=True)
'''


notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# Cross-plane token residual screen\n",
                    "Fixed five-seed private validation; organizer gold is evaluation-only; no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run",
         "metadata": {}, "outputs": [], "source": SOURCE.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "crossplane-token-residual.ipynb").write_text(json.dumps(notebook, indent=1))

metadata = {
    "id": "sarthaksharma14/rsna-knee-crossplane-token-residual",
    "title": "RSNA Knee Cross-plane Token Residual",
    "code_file": "crossplane-token-residual.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels"],
    "kernel_sources": ["sarthaksharma14/rsna-knee-convnext-reader-feature-cache",
                       "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit",
                       "sarthaksharma14/rsna-knee-multireader-nested-blend-screen"],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
