"""Private gold-only screen of an attributed RSNA-trained ConvNeXtV2 backbone."""

import copy
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
SEEDS = (20261007, 20261008, 20261009)
N_STUDIES, N_SLOT, N_DEPTH, FEATURE_DIM = 4407, 6, 10, 768
HEAD_EPOCHS = 45
CONTROL_GOLD_AUC = 0.860457
CHECKPOINT_NAME = "hunter_pool_v1_best.pt"
CHECKPOINT_SHA256 = "4680dff05e3556eebbF07e84ca824ab6f1fe8db442dc9cc3948e18d226f008c3".lower()
CACHE_SHA256 = "490e5ccc07143699d05a9eb3ef9583e08795f1c2bacc6d9cf2bd0a9268158526"
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
        raise RuntimeError(f"expected exactly one {name}, found {matches}")
    return matches[0]


def competition_root():
    for root in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                 Path("/kaggle/input/rsna-knee-abnormality-detection")):
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


def load_backbone(checkpoint_path):
    import timm

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "head.weight" not in state:
        raise RuntimeError("unexpected ConvNeXtV2 checkpoint container")
    if tuple(state["head.weight"].shape) != (12, 1536):
        raise RuntimeError(f"unexpected published pool head: {state['head.weight'].shape}")
    backbone_state = {key[2:]: value for key, value in state.items()
                      if key.startswith("0.")}
    model = timm.create_model("convnextv2_tiny", pretrained=False,
                              num_classes=0, global_pool="")
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    allowed_missing = {"head.fc.weight", "head.fc.bias"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")
    if int(model.num_features) != FEATURE_DIM:
        raise RuntimeError(f"feature dimension drift: {model.num_features}")
    model = model.to(DEVICE).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del state, backbone_state
    return model


class H5StudyRows(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = str(path)
        self.handle = None

    def __len__(self):
        return N_STUDIES

    def __getitem__(self, row):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r", swmr=True)
        return row, self.handle["pixels"][row], self.handle["masks"][row]


@torch.no_grad()
def encode_cache(h5_path, model):
    output_path = Path("/kaggle/working/convnextv2_domain_features.npy")
    features = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16,
        shape=(N_STUDIES, N_SLOT, N_DEPTH, FEATURE_DIM))
    mean = torch.tensor([.485, .456, .406], device=DEVICE)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=DEVICE)[None, :, None, None]
    loader = DataLoader(H5StudyRows(h5_path), batch_size=4, shuffle=False,
                        num_workers=4, prefetch_factor=1, persistent_workers=True,
                        pin_memory=True)
    completed = 0
    for rows, studies, masks in loader:
        batch = len(rows)
        # Twelve physical slices yield ten adjacent 2.5D triplets.
        windows = torch.stack((studies[:, :, :-2], studies[:, :, 1:-1],
                               studies[:, :, 2:]), dim=3)
        token_mask = masks.bool()[:, :, None].expand(-1, -1, N_DEPTH)
        images = windows[token_mask]
        batch_features = []
        for start in range(0, len(images), 32):
            image = images[start:start + 32].to(
                DEVICE, non_blocking=True).float().div_(255.0)
            image = (image - mean) / std
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                fmap = model(image.contiguous(memory_format=torch.channels_last))
                encoded = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
            batch_features.append(encoded.float().cpu())
        encoded = torch.cat(batch_features).numpy().astype(np.float16)
        locations = token_mask.nonzero(as_tuple=False).numpy()
        row_values = rows.numpy()
        for location, value in zip(locations, encoded):
            batch_row, slot, depth = location
            features[row_values[batch_row], slot, depth] = value
        completed += batch
        if completed % 100 <= batch:
            features.flush()
            log(f"encoded {completed}/{N_STUDIES} studies")
    features.flush()
    del features
    gc.collect()
    return output_path


class StudyTransformerHead(nn.Module):
    """Public reader contract: slot/depth tokens, transformer, target attention."""

    def __init__(self, dim=384, depth=2, drop=.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(FEATURE_DIM, dim),
                                  nn.LayerNorm(dim))
        self.slot = nn.Embedding(N_SLOT, dim)
        self.position = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=.1,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth,
                                                  enable_nested_tensor=False)
        self.attention = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(),
                                       nn.Dropout(drop), nn.Linear(256, len(TARGETS)))
        self.classifier_weight = nn.Parameter(torch.randn(len(TARGETS), dim) * .02)
        self.classifier_bias = nn.Parameter(torch.zeros(len(TARGETS)))
        position = torch.linspace(1 / 11, 10 / 11, N_DEPTH)
        self.register_buffer("depth_position", position)

    def forward(self, feature, slot_mask):
        batch = feature.shape[0]
        slots = torch.arange(N_SLOT, device=feature.device)[None, :, None]
        positions = self.depth_position[None, None, :, None].expand(batch, N_SLOT, -1, -1)
        hidden = self.proj(feature) + self.slot(slots) + self.position(positions)
        hidden = hidden.reshape(batch, N_SLOT * N_DEPTH, -1)
        padding = ~slot_mask.bool()[:, :, None].expand(-1, -1, N_DEPTH).reshape(batch, -1)
        padding = padding & ~padding.all(1, keepdim=True)
        hidden = self.transformer(hidden, src_key_padding_mask=padding)
        score = self.attention(hidden).float().masked_fill(padding[:, :, None], -1e4)
        weights = score.softmax(1)
        pooled = torch.einsum("btc,btd->bcd", weights.to(hidden.dtype), hidden)
        return ((pooled * self.classifier_weight).sum(-1) + self.classifier_bias)


@torch.no_grad()
def predict(model, features, masks, indices):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(np.asarray(features[indices], np.float32)),
                            torch.from_numpy(np.asarray(masks[indices], np.float32)))
    outputs = []
    for feature, mask in DataLoader(dataset, batch_size=128, shuffle=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            outputs.append(torch.sigmoid(model(feature.to(DEVICE), mask.to(DEVICE)))
                           .float().cpu().numpy())
    return np.concatenate(outputs)


def train_head(features, masks, labels, fitting, validation, seed):
    seed_all(seed)
    model = StudyTransformerHead().to(DEVICE)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_EPOCHS)
    dataset = TensorDataset(torch.from_numpy(np.asarray(features[fitting], np.float32)),
                            torch.from_numpy(np.asarray(masks[fitting], np.float32)),
                            torch.from_numpy(labels[fitting]))
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    weak_truth = (labels[validation] >= .5).astype(np.int8)
    best_auc, best_state, history, step = -np.inf, None, [], 0
    for epoch in range(HEAD_EPOCHS):
        model.train()
        running = 0.0
        for feature, mask, target in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits = model(feature.to(DEVICE), mask.to(DEVICE))
                loss = F.binary_cross_entropy_with_logits(logits.float(), target.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            step += 1
            decay = min(.998, (1 + step) / (10 + step))
            with torch.no_grad():
                for target_parameter, source_parameter in zip(ema.parameters(), model.parameters()):
                    target_parameter.lerp_(source_parameter, 1 - decay)
            running += float(loss.detach()) * len(feature)
        scheduler.step()
        weak_prediction = predict(ema, features, masks, validation)
        weak_auc = macro_auc(weak_truth, weak_prediction)
        history.append({"epoch": epoch + 1, "loss": running / len(dataset),
                        "weak_validation_auc": weak_auc})
        if weak_auc > best_auc:
            best_auc = weak_auc
            best_state = {key: value.detach().cpu().clone()
                          for key, value in ema.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            log(f"seed {seed} epoch {epoch + 1}: weak AUC {weak_auc:.6f}")
    ema.load_state_dict(best_state)
    return ema, best_state, best_auc, history


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU required")
    checkpoint = unique_input(CHECKPOINT_NAME)
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint hash drift: {checkpoint_hash}")
    h5_path = unique_input("train_288x12.h5")
    with h5py.File(h5_path, "r") as handle:
        if handle["pixels"].shape != (N_STUDIES, N_SLOT, 12, 288, 288):
            raise RuntimeError(f"cache shape drift: {handle['pixels'].shape}")
        masks_cache_order = np.asarray(handle["masks"], np.float32)

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
    if len(studies) != N_STUDIES or not studies.StudyInstanceUID.is_unique:
        raise RuntimeError("cache study manifest drift")
    order = pd.Series(np.arange(len(studies)), index=studies.StudyInstanceUID)
    reorder = order.loc[train.StudyInstanceUID].to_numpy()
    masks = masks_cache_order[reorder]

    backbone = load_backbone(checkpoint)
    feature_path = encode_cache(h5_path, backbone)
    del backbone
    torch.cuda.empty_cache()
    features = np.load(feature_path, mmap_mode="r")[reorder]

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
        rows.append({"seed": seed, "best_weak_auc": weak_auc,
                     "gold_auc": gold_auc, "history": history})
        predictions.append(gold_prediction)
        states.append(state)
        log(f"seed {seed}: gold AUC {gold_auc:.6f}")

    ensemble_auc = macro_auc(gold_truth, np.mean(predictions, axis=0))
    heads_path = Path("/kaggle/working/convnextv2_domain_heads.pt")
    torch.save({"states": states, "seeds": SEEDS, "targets": TARGETS}, heads_path)
    report = {
        "contract": "private one-fold domain-backbone screen; no submission",
        "candidate": "frozen public RSNA ConvNeXtV2-Tiny backbone plus study Transformer head",
        "public_model": "yaolinghan/rsna-knee-pool-models/PyTorch/convnextv2-pool/1",
        "public_model_license": "Apache-2.0",
        "public_checkpoint": CHECKPOINT_NAME,
        "public_checkpoint_sha256": checkpoint_hash,
        "cache_h5_sha256": CACHE_SHA256,
        "feature_sha256": sha256(feature_path),
        "heads_sha256": sha256(heads_path),
        "label_source": "stevenleehans llm_labels_v4_blend.csv",
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
    Path("/kaggle/working/convnextv2_domain_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


main()
