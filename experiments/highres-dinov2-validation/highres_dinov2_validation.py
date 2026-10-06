"""Private leakage-safe screen of frozen DINOv2-S on the 288px/12-slice cache."""

import gc
import hashlib
import json
import random
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
SEEDS = (20261005, 20261006, 20261007)
N_SLOT, N_DEPTH, FEATURE_DIM = 6, 4, 768
HEAD_EPOCHS = 50
CONTROL_GOLD_AUC = 0.860457
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START = time.time()


def log(message):
    print(f"[{time.time() - START:7.1f}s] {message}", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def unique_input(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0]


def competition_root():
    candidates = (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                  Path("/kaggle/input/rsna-knee-abnormality-detection"))
    for root in candidates:
        if (root / "train.csv").is_file():
            return root
    raise FileNotFoundError("competition train.csv not mounted")


def macro_auc(truth, prediction):
    return float(np.mean([roc_auc_score(truth[:, i], prediction[:, i])
                          for i in range(truth.shape[1])]))


def report_groups(series):
    normalized = (series.fillna("").astype(str).str.lower()
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in normalized])


def load_encoder():
    from transformers import AutoModel
    roots = sorted(set(path.parent for path in Path("/kaggle/input").rglob("config.json")
                       if "dinov2" in str(path).lower()))
    if len(roots) != 1:
        raise RuntimeError(f"expected one DINOv2 model, found {roots}")
    model = AutoModel.from_pretrained(str(roots[0])).to(DEVICE).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, str(roots[0])


class H5StudyRows(torch.utils.data.Dataset):
    """Open one read-only HDF5 handle per worker and stream contiguous studies."""

    def __init__(self, path):
        self.path = str(path)
        self.handle = None

    def __len__(self):
        return 4407

    def __getitem__(self, row):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r", swmr=True)
        return row, self.handle["pixels"][row], self.handle["masks"][row]


@torch.no_grad()
def encode_cache(h5_path, model):
    output_path = Path("/kaggle/working/highres_dinov2_features.npy")
    features = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16,
        shape=(4407, N_SLOT, N_DEPTH, FEATURE_DIM))
    mean = torch.tensor([.485, .456, .406], device=DEVICE)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=DEVICE)[None, :, None, None]
    with h5py.File(h5_path, "r") as handle:
        if handle["pixels"].shape != (4407, 6, 12, 288, 288):
            raise RuntimeError(f"high-resolution cache shape drift: {handle['pixels'].shape}")
    loader = DataLoader(H5StudyRows(h5_path), batch_size=4, shuffle=False,
                        num_workers=4, prefetch_factor=1, persistent_workers=True,
                        pin_memory=True)
    completed = 0
    for rows, studies, masks in loader:
        batch = len(rows)
        images = studies.reshape(batch, N_SLOT, N_DEPTH, 3, 288, 288)
        token_mask = masks.bool()[:, :, None].expand(-1, -1, N_DEPTH)
        locations = token_mask.nonzero(as_tuple=False)
        images = images[token_mask]
        batch_features = []
        for start in range(0, len(images), 32):
            image = images[start:start + 32].to(
                DEVICE, non_blocking=True).float().div_(255.0)
            image = (image - mean) / std
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                hidden = model(pixel_values=image, interpolate_pos_encoding=True).last_hidden_state
                encoded = torch.cat((hidden[:, 0], hidden[:, 1:].mean(1)), dim=1)
            batch_features.append(encoded.float().cpu())
        encoded = torch.cat(batch_features).numpy().astype(np.float16)
        row_values = rows.numpy()
        for location, value in zip(locations.numpy(), encoded):
            batch_row, slot, depth = location
            features[row_values[batch_row], slot, depth] = value
        completed += batch
        if completed % 100 <= batch:
            features.flush()
            log(f"encoded {completed}/{len(features)} studies")
    features.flush()
    del features
    gc.collect()
    return output_path


class TargetDepthHead(nn.Module):
    def __init__(self, hidden=320):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, hidden),
                                  nn.GELU(), nn.Dropout(.15))
        self.slot = nn.Parameter(torch.randn(N_SLOT, hidden) * .02)
        self.depth = nn.Parameter(torch.randn(N_DEPTH, hidden) * .02)
        self.query = nn.Parameter(torch.randn(len(TARGETS), hidden) * .02)
        self.out = nn.Linear(hidden, len(TARGETS))
        self.dropout = nn.Dropout(.25)

    def forward(self, feature, mask):
        hidden = self.proj(feature)
        hidden = hidden + self.slot[:, None] + self.depth[None, :]
        hidden = hidden.flatten(1, 2)
        token_mask = mask[:, :, None].expand(-1, -1, N_DEPTH).flatten(1)
        score = torch.einsum("bnh,th->btn", hidden, self.query) / hidden.shape[-1] ** .5
        score = score.masked_fill(token_mask[:, None] < .5, -1e4).softmax(-1)
        pooled = self.dropout(torch.einsum("btn,bnh->bth", score, hidden))
        return (pooled * self.out.weight[None]).sum(-1) + self.out.bias


@torch.no_grad()
def predict(model, features, masks, indices):
    model.eval()
    outputs = []
    dataset = TensorDataset(torch.from_numpy(np.asarray(features[indices], np.float32)),
                            torch.from_numpy(np.asarray(masks[indices], np.float32)))
    for feature, mask in DataLoader(dataset, batch_size=128, shuffle=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            outputs.append(torch.sigmoid(model(feature.to(DEVICE), mask.to(DEVICE))).cpu().numpy())
    return np.concatenate(outputs)


def train_head(features, masks, labels, fitting, validation, seed):
    seed_all(seed)
    model = TargetDepthHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_EPOCHS)
    dataset = TensorDataset(torch.from_numpy(np.asarray(features[fitting], np.float32)),
                            torch.from_numpy(np.asarray(masks[fitting], np.float32)),
                            torch.from_numpy(labels[fitting]))
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_auc, best_state, history = -np.inf, None, []
    weak_truth = (labels[validation] >= .5).astype(np.int8)
    for epoch in range(HEAD_EPOCHS):
        model.train()
        running = 0.0
        for feature, mask, target in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(feature.to(DEVICE), mask.to(DEVICE)), target.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(feature)
        scheduler.step()
        weak_prediction = predict(model, features, masks, validation)
        weak_auc = macro_auc(weak_truth, weak_prediction)
        history.append({"epoch": epoch + 1, "loss": running / len(dataset),
                        "weak_validation_auc": weak_auc})
        if weak_auc > best_auc:
            best_auc = weak_auc
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            log(f"seed {seed} epoch {epoch + 1}: weak AUC {weak_auc:.6f}")
    model.load_state_dict(best_state)
    return model, best_state, best_auc, history


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU required")
    seed_all(SEEDS[0])
    train = pd.read_csv(competition_root() / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError("organizer gold count drift")
    labels_source = pd.read_csv(unique_input("llm_labels_v4_blend.csv"),
                                dtype={"StudyInstanceUID": str})
    aligned = train[["StudyInstanceUID"]].merge(labels_source, on="StudyInstanceUID",
                                                   how="left", validate="one_to_one")
    labels = aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all():
        raise RuntimeError("Steven v4 labels incomplete")

    studies = pd.read_csv(unique_input("highres_cache_studies.csv"),
                          dtype={"StudyInstanceUID": str})
    if len(studies) != 4407 or not studies.StudyInstanceUID.is_unique:
        raise RuntimeError("cache study manifest drift")
    order = pd.Series(np.arange(len(studies)), index=studies.StudyInstanceUID)
    if not set(train.StudyInstanceUID).issubset(order.index):
        raise RuntimeError("cache misses training studies")
    reorder = order.loc[train.StudyInstanceUID].to_numpy()
    h5_path = unique_input("train_288x12.h5")
    with h5py.File(h5_path, "r") as handle:
        masks = np.asarray(handle["masks"], np.float32)[reorder]

    encoder, encoder_root = load_encoder()
    feature_path = encode_cache(h5_path, encoder)
    del encoder
    torch.cuda.empty_cache()
    raw_features = np.load(feature_path, mmap_mode="r")
    features = raw_features[reorder]

    non_gold = np.flatnonzero(~gold)
    groups = report_groups(train.Report)
    train_local, validation_local = list(GroupKFold(5).split(
        non_gold, groups=groups[non_gold]))[0]
    fitting, validation = non_gold[train_local], non_gold[validation_local]
    if set(groups[fitting]).intersection(groups[validation]):
        raise RuntimeError("report-group leakage")
    gold_index = np.flatnonzero(gold)
    gold_truth = train.loc[gold, TARGETS].to_numpy(np.float32)

    rows, predictions, states = [], [], []
    for seed in SEEDS:
        model, state, weak_auc, history = train_head(
            features, masks, labels, fitting, validation, seed)
        gold_prediction = predict(model, features, masks, gold_index)
        gold_auc = macro_auc(gold_truth, gold_prediction)
        rows.append({"seed": seed, "best_weak_auc": weak_auc, "gold_auc": gold_auc,
                     "history": history})
        predictions.append(gold_prediction)
        states.append(state)
        log(f"seed {seed}: gold AUC {gold_auc:.6f}")

    ensemble_prediction = np.mean(predictions, axis=0)
    ensemble_auc = macro_auc(gold_truth, ensemble_prediction)
    checkpoint = Path("/kaggle/working/highres_dinov2_heads.pt")
    torch.save({"states": states, "seeds": SEEDS, "targets": TARGETS,
                "encoder_root": encoder_root}, checkpoint)
    report = {
        "contract": "private one-fold high-resolution screen; no submission",
        "candidate": "frozen DINOv2-S 288px, 12 slices, six slots, target depth attention",
        "label_source": "stevenleehans llm_labels_v4_blend.csv",
        "attribution": ["metaresearch/dinov2/PyTorch/small/1",
                        "stevenleehans/rsna-knee-llm-report-labels",
                        "stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache"],
        "cache_h5_sha256": "490e5ccc07143699d05a9eb3ef9583e08795f1c2bacc6d9cf2bd0a9268158526",
        "feature_sha256": sha256(feature_path),
        "checkpoint_sha256": sha256(checkpoint),
        "fit_studies": int(len(fitting)),
        "weak_validation_studies": int(len(validation)),
        "gold_fit_count": 0,
        "gold_evaluation_count": 58,
        "seeds": rows,
        "ensemble_gold_auc": ensemble_auc,
        "prior_private_control_auc": CONTROL_GOLD_AUC,
        "delta_vs_control": ensemble_auc - CONTROL_GOLD_AUC,
        "screen_passed": bool(ensemble_auc >= CONTROL_GOLD_AUC + .005 and
                              sum(row["gold_auc"] > CONTROL_GOLD_AUC for row in rows) >= 2),
        "submission_created": False,
    }
    Path("/kaggle/working/highres_dinov2_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


main()
