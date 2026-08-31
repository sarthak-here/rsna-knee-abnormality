"""Private held-out validation for a materially distinct MedicalNet 3D encoder.

The 58 organizer-labelled studies are never used for fitting. A frozen MedicalNet
3D ResNet-50 extracts multi-planar volumetric features from the remaining 4,349
studies, and lightweight heads are trained only on public report-derived labels.
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
PLANES = ("Sagittal", "Coronal", "Axial")
SEEDS = (20260831, 20260832, 20260833, 20260834, 20260835)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:7.1f}s] {message}", flush=True)


def locate_competition() -> Path:
    for path in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (path / "train.csv").is_file():
            return path
    raise FileNotFoundError("RSNA knee competition data is not attached")


def locate_file(name: str, required_fragment: str) -> Path:
    matches = [p for p in Path("/kaggle/input").rglob(name) if required_fragment in str(p)]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name} under {required_fragment}, found {matches}")
    return matches[0]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedicalNetResNet50(nn.Module):
    """MedicalNet ResNet-50 backbone, shortcut type B, without segmentation head."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.pool(x).flatten(1)


def load_backbone(weight_path: Path) -> nn.Module:
    model = MedicalNetResNet50()
    payload = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("conv_seg"):
            continue
        if key in model.state_dict() and model.state_dict()[key].shape == value.shape:
            cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    backbone_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if backbone_missing or unexpected:
        raise RuntimeError(f"MedicalNet checkpoint mismatch: missing={backbone_missing}, unexpected={unexpected}")
    log(f"loaded {len(cleaned)} MedicalNet tensors from {weight_path.name}")
    return model.eval().to(DEVICE)


def natural_key(path: Path):
    import re
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def read_series(series_dir: Path, slices: int = 16, size: int = 96) -> torch.Tensor:
    files = sorted(series_dir.glob("*.dcm"), key=natural_key)
    if not files:
        return torch.zeros((slices, size, size), dtype=torch.float32)
    lo, hi = int(0.15 * (len(files) - 1)), int(0.85 * (len(files) - 1))
    indices = np.linspace(lo, max(lo, hi), slices).round().astype(int)
    images = []
    for index in indices:
        try:
            ds = pydicom.dcmread(files[int(index)], force=True)
            image = ds.pixel_array.astype(np.float32)
            image = image * float(getattr(ds, "RescaleSlope", 1) or 1)
            image += float(getattr(ds, "RescaleIntercept", 0) or 0)
            low, high = np.percentile(image, (1, 99))
            image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
            side = min(image.shape)
            y0, x0 = (image.shape[0] - side) // 2, (image.shape[1] - side) // 2
            image = image[y0:y0 + side, x0:x0 + side]
            tensor = torch.from_numpy(image)[None, None]
            tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
        except Exception:
            tensor = torch.zeros((size, size), dtype=torch.float32)
        images.append(tensor)
    return torch.stack(images)


def choose_series(series: pd.DataFrame) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for study, group in series.groupby("StudyInstanceUID", sort=False):
        mapping = {}
        for plane in PLANES:
            candidates = group[
                (group["Anatomical_Plane"] == plane)
                & (group["Fluid_Sensitive"] == 1)
                & (group["Fat_Suppression"] == 1)
            ]
            if candidates.empty:
                candidates = group[group["Anatomical_Plane"] == plane]
            if not candidates.empty:
                mapping[plane] = str(candidates.iloc[0]["SeriesInstanceUID"])
        selected[str(study)] = mapping
    return selected


@torch.inference_mode()
def extract_features(ids, selected, series_root: Path, model: nn.Module, cache_path: Path):
    if cache_path.is_file():
        cache = np.load(cache_path, allow_pickle=False)
        if cache["ids"].astype(str).tolist() == list(ids):
            log(f"using cached features from {cache_path.name}")
            return cache["features"].astype(np.float32)
    features = np.zeros((len(ids), 2048), dtype=np.float32)
    for row, study in enumerate(ids):
        volumes = []
        for plane in PLANES:
            series_uid = selected.get(study, {}).get(plane)
            path = series_root / study / series_uid if series_uid else Path("/__missing__")
            volumes.append(read_series(path))
        volume = torch.cat(volumes, dim=0)[None, None].to(device=DEVICE, dtype=torch.float32)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            features[row] = model(volume).float().cpu().numpy()[0]
        if (row + 1) % 100 == 0 or row + 1 == len(ids):
            log(f"features {row + 1}/{len(ids)}")
    np.savez_compressed(cache_path, ids=np.asarray(ids), features=features)
    return features


class Head(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.25), nn.Linear(512, 12)
        )

    def forward(self, x):
        return self.net(x)


def train_head(features, labels, gold_features, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    order = np.random.default_rng(seed).permutation(len(features))
    cut = int(0.85 * len(order))
    train_idx, valid_idx = order[:cut], order[cut:]
    mean = features[train_idx].mean(0, keepdims=True)
    std = features[train_idx].std(0, keepdims=True).clip(1e-5)
    x = torch.from_numpy(((features - mean) / std).astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    gx = torch.from_numpy(((gold_features - mean) / std).astype(np.float32)).to(DEVICE)
    model = Head(features.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    best_state, best_loss, stale = None, float("inf"), 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(80):
        model.train()
        for batch in torch.utils.data.DataLoader(train_idx.tolist(), batch_size=128, shuffle=True, generator=generator):
            logits = model(x[batch].to(DEVICE))
            loss = F.binary_cross_entropy_with_logits(logits, y[batch].to(DEVICE))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = F.binary_cross_entropy_with_logits(
                model(x[valid_idx].to(DEVICE)), y[valid_idx].to(DEVICE)
            ).item()
        if value < best_loss - 1e-5:
            best_loss, stale = value, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(model(gx)).cpu().numpy()
    log(f"seed={seed} weak-valid BCE={best_loss:.5f} epochs={epoch + 1}")
    return prediction


def rank_columns(values):
    return pd.DataFrame(values).rank(method="average", pct=True).to_numpy(np.float64)


def macro_auc(y, pred):
    per_target = [roc_auc_score(y[:, j], pred[:, j]) for j in range(y.shape[1])]
    return float(np.mean(per_target)), dict(zip(TARGETS, map(float, per_target)))


def bootstrap_delta(y, baseline, candidate, draws=2000):
    rng = np.random.default_rng(20260831)
    deltas = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        usable = [j for j in range(y.shape[1]) if np.unique(y[idx, j]).size == 2]
        if not usable:
            continue
        b = np.mean([roc_auc_score(y[idx, j], baseline[idx, j]) for j in usable])
        c = np.mean([roc_auc_score(y[idx, j], candidate[idx, j]) for j in usable])
        deltas.append(c - b)
    values = np.asarray(deltas)
    return {
        "draws": int(len(values)), "mean": float(values.mean()),
        "p_improve": float(np.mean(values > 0)),
        "q025": float(np.quantile(values, 0.025)), "q975": float(np.quantile(values, 0.975)),
    }


def load_public_screening_control(gold_ids):
    """Load public RadImageNet OOF as a screening control, not the locked baseline."""
    rad_path = locate_file("v52_oof.csv", "rsna-knee-e9-radimagenet-heads-v15")
    rad = pd.read_csv(rad_path, dtype={"StudyInstanceUID": str})
    ids = pd.DataFrame({"StudyInstanceUID": gold_ids})
    r = ids.merge(rad[["StudyInstanceUID", *TARGETS]], on="StudyInstanceUID", how="left", validate="one_to_one")
    if r[TARGETS].isna().any().any():
        raise RuntimeError("public RadImageNet OOF alignment failed")
    return rank_columns(r[TARGETS].to_numpy())


def main():
    comp = locate_competition()
    train = pd.read_csv(comp / "train.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(comp / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    labels_path = locate_file("llm_labels_v4_blend.csv", "rsna-knee-llm-report-labels")
    weak = pd.read_csv(labels_path, dtype={"StudyInstanceUID": str})
    gold_mask = train[TARGETS].notna().all(axis=1)
    gold = train.loc[gold_mask, ["StudyInstanceUID", *TARGETS]].copy()
    if len(gold) != 58:
        raise RuntimeError(f"expected 58 gold studies, got {len(gold)}")
    gold_ids = gold["StudyInstanceUID"].tolist()
    weak = weak[~weak["StudyInstanceUID"].isin(gold_ids)].copy()
    weak = weak[weak["StudyInstanceUID"].isin(train["StudyInstanceUID"])].reset_index(drop=True)
    if len(weak) != 4349:
        raise RuntimeError(f"expected 4349 weak-only studies, got {len(weak)}")
    overlap = set(weak["StudyInstanceUID"]) & set(gold_ids)
    if overlap:
        raise RuntimeError(f"gold leakage detected: {len(overlap)} studies")
    log("leakage gate passed: 4,349 weak-only train studies; 58 untouched gold studies")

    selected = choose_series(series)
    weight_path = locate_file("resnet_50_23dataset.pth", "rsna-medicalnet-resnet50-weights")
    model = load_backbone(weight_path)
    weak_ids = weak["StudyInstanceUID"].tolist()
    all_ids = weak_ids + gold_ids
    features = extract_features(
        all_ids, selected, comp / "train_series", model, Path("/kaggle/working/medicalnet_features.npz")
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    weak_features, gold_features = features[: len(weak_ids)], features[len(weak_ids):]
    weak_y = weak[TARGETS].to_numpy(np.float32)
    gold_y = gold[TARGETS].to_numpy(np.int64)
    seed_predictions = np.stack(
        [train_head(weak_features, weak_y, gold_features, seed) for seed in SEEDS]
    )
    medicalnet = rank_columns(seed_predictions.mean(0))
    control = load_public_screening_control(gold_ids)
    candidate = rank_columns(0.8 * control + 0.2 * medicalnet)
    control_score, control_targets = macro_auc(gold_y, control)
    medical_score, medical_targets = macro_auc(gold_y, medicalnet)
    candidate_score, candidate_targets = macro_auc(gold_y, candidate)
    seed_deltas = []
    for prediction in seed_predictions:
        score, _ = macro_auc(gold_y, rank_columns(0.8 * control + 0.2 * rank_columns(prediction)))
        seed_deltas.append(float(score - control_score))
    report = {
        "experiment": "MedicalNet 3D ResNet-50 frozen volumetric probe",
        "provenance": {
            "backbone": "Tencent MedicalNet / Med3D ResNet-50 pretrained on 23 datasets",
            "weights": "royceplayz/rsna-medicalnet-resnet50-weights/resnet_50_23dataset.pth",
            "weak_labels": "stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
            "gold_rows_used_for_training": 0,
        },
        "screening_control": "public RadImageNet E9 OOF; not the locked 0.936 ensemble",
        "screening_control_macro_auc": control_score,
        "medicalnet_macro_auc": medical_score,
        "fixed_20pct_blend_macro_auc": candidate_score,
        "fixed_20pct_delta_vs_control": candidate_score - control_score,
        "seed_blend_deltas": seed_deltas,
        "positive_seed_count": int(np.sum(np.asarray(seed_deltas) > 0)),
        "bootstrap": bootstrap_delta(gold_y, control, candidate),
        "per_target": {
            "screening_control": control_targets, "medicalnet": medical_targets, "candidate": candidate_targets,
        },
    }
    report["passes_screening_gate"] = bool(
        report["fixed_20pct_delta_vs_control"] >= 0.005
        and report["positive_seed_count"] >= 4
        and report["bootstrap"]["p_improve"] >= 0.95
        and report["bootstrap"]["q025"] > 0
    )
    Path("/kaggle/working/medicalnet_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        "/kaggle/working/medicalnet_gold_predictions.npz",
        ids=np.asarray(gold_ids), screening_control=control, medicalnet=medicalnet,
        candidate=candidate, seed_predictions=seed_predictions,
    )
    print(json.dumps(report, indent=2), flush=True)
    print("PASS_SCREENING_GATE" if report["passes_screening_gate"] else "REJECT_CANDIDATE", flush=True)


if __name__ == "__main__":
    main()
