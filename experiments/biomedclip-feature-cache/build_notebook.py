"""Build the private frozen BiomedCLIP physical feature-cache notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = r'''from pathlib import Path
import hashlib, json, time
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm

H5_SHA = "490e5ccc07143699d05a9eb3ef9583e08795f1c2bacc6d9cf2bd0a9268158526"
STUDIES_SHA = "efc63afee30345f94a916f98ba6b30f266fccfe2976b8475fa86653b2c510513"
MODEL_SHA = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
CONFIG_SHA = "9a41f334a8c444678772c0ebb9ab854c97ab350bced3a17b803e258d39c23dc0"
EXPECTED_SHAPE = (4407, 6, 4, 512)
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working")

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def unique_input(name):
    hits = [path for path in Path("/kaggle/input").rglob(name) if path.is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(hits)}: {hits[:8]}")
    return hits[0]

if DEVICE.type != "cuda":
    raise RuntimeError("GPU required for BiomedCLIP feature extraction")

started = time.time()
h5_path = unique_input("train_288x12.h5")
studies_path = unique_input("highres_cache_studies.csv")
model_path = unique_input("open_clip_pytorch_model.bin")
config_path = unique_input("open_clip_config.json")
if h5_path.stat().st_size != 17051786047 or sha256(h5_path) != H5_SHA:
    raise RuntimeError("physical cache identity drift")
if sha256(studies_path) != STUDIES_SHA:
    raise RuntimeError("physical study manifest drift")
if sha256(model_path) != MODEL_SHA or sha256(config_path) != CONFIG_SHA:
    raise RuntimeError("BiomedCLIP asset identity drift")

ids = pd.read_csv(studies_path, dtype=str).StudyInstanceUID.astype(str).to_numpy()
if len(ids) != 4407 or len(set(ids)) != 4407:
    raise RuntimeError("study identity/count drift")

checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
trunk_state = {key.removeprefix("visual.trunk."): value
               for key, value in checkpoint.items() if key.startswith("visual.trunk.")}
if len(trunk_state) != 150 or "visual.head.proj.weight" not in checkpoint:
    raise RuntimeError("BiomedCLIP state-dict schema drift")
encoder = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
loaded = encoder.load_state_dict(trunk_state, strict=True)
if loaded.missing_keys or loaded.unexpected_keys:
    raise RuntimeError(f"strict visual load failed: {loaded}")
projection = checkpoint["visual.head.proj.weight"].float().to(DEVICE)
if tuple(projection.shape) != (512, 768):
    raise RuntimeError("BiomedCLIP projection shape drift")
encoder.eval().to(DEVICE)
for parameter in encoder.parameters():
    parameter.requires_grad_(False)
del checkpoint, trunk_state

features = np.zeros(EXPECTED_SHAPE, np.float16)
all_masks = np.zeros((4407, 6), np.uint8)
mean, std = MEAN.to(DEVICE), STD.to(DEVICE)

@torch.inference_mode()
def encode(images):
    tensor = torch.from_numpy(images).to(DEVICE).float().div_(255.0)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear",
                           align_corners=False, antialias=True)
    tensor = (tensor - mean) / std
    with torch.autocast("cuda", dtype=torch.float16):
        embedding = encoder(tensor).float() @ projection.T
        embedding = F.normalize(embedding, dim=1)
    return embedding.cpu().numpy().astype(np.float16)

with h5py.File(h5_path, "r") as handle:
    pixels, masks = handle["pixels"], handle["masks"]
    if pixels.shape != (4407, 6, 12, 288, 288) or masks.shape != (4407, 6):
        raise RuntimeError("physical cache tensor drift")
    for start in range(0, 4407, 8):
        stop = min(start + 8, 4407)
        block = pixels[start:stop]
        block_masks = masks[start:stop].astype(np.uint8)
        all_masks[start:stop] = block_masks
        images, assignments = [], []
        for local in range(stop - start):
            for slot in range(6):
                if not block_masks[local, slot]:
                    continue
                for window in range(4):
                    images.append(block[local, slot, window * 3:(window + 1) * 3])
                    assignments.append((start + local, slot, window))
        if images:
            array = np.stack(images)
            encoded = []
            for offset in range(0, len(array), 64):
                encoded.append(encode(array[offset:offset + 64]))
            for vector, (row, slot, window) in zip(np.concatenate(encoded), assignments):
                features[row, slot, window] = vector
        if start % 200 == 0:
            print(f"BiomedCLIP encoded {start}/4407", flush=True)

if features.shape != EXPECTED_SHAPE or not np.isfinite(features).all():
    raise RuntimeError("feature output drift")
if all_masks.shape != (4407, 6) or not np.isin(all_masks, (0, 1)).all():
    raise RuntimeError("mask output drift")
if (all_masks.sum(1) == 0).any():
    raise RuntimeError("study without occupied slot")
for row in range(4407):
    for slot in range(6):
        nonzero = bool(np.any(features[row, slot]))
        if nonzero != bool(all_masks[row, slot]):
            raise RuntimeError(f"feature/mask mismatch at {row},{slot}")

feature_path = OUT / "biomedclip_physical_features.npz"
safe_ids = ids.astype("U")
if safe_ids.dtype.kind != "U":
    raise RuntimeError("study IDs must be pickle-free Unicode")
np.savez_compressed(feature_path, ids=safe_ids, features=features, masks=all_masks,
                    model_sha256=MODEL_SHA, config_sha256=CONFIG_SHA,
                    h5_sha256=H5_SHA, studies_sha256=STUDIES_SHA)
receipt = {
    "status": "COMPLETE", "contract": "frozen BiomedCLIP physical feature cache; no labels; no submission",
    "study_count": 4407, "unique_studies": int(len(set(ids))),
    "ids_dtype": str(safe_ids.dtype), "pickle_free": True,
    "feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
    "occupied_slots": int(all_masks.sum()), "finite": bool(np.isfinite(features).all()),
    "model_sha256": MODEL_SHA, "config_sha256": CONFIG_SHA,
    "h5_sha256": H5_SHA, "studies_sha256": STUDIES_SHA,
    "output_sha256": sha256(feature_path), "runtime_seconds": time.time() - started,
    "submission_created": False,
}
(OUT / "biomedclip_feature_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
for csv_path in OUT.glob("*.csv"):
    csv_path.unlink()
if list(OUT.glob("*.csv")):
    raise RuntimeError("forbidden CSV artifact")
print(json.dumps(receipt, indent=2), flush=True)
'''

compile(SOURCE, "biomedclip-feature-cache", "exec")
notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# Frozen BiomedCLIP physical feature cache\n",
                    "MIT-licensed medical image representation; no labels and no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run", "metadata": {},
         "outputs": [], "source": SOURCE.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(ROOT / "biomedclip-feature-cache.ipynb").write_text(json.dumps(notebook, indent=1))
metadata = {
    "id": "sarthaksharma14/rsna-knee-biomedclip-feature-cache",
    "title": "RSNA Knee BiomedCLIP Feature Cache",
    "code_file": "biomedclip-feature-cache.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "competition_sources": [],
    "dataset_sources": ["kayven/microsoft-biomedclip-official-visual-checkpoint"],
    "kernel_sources": ["sarthaksharma14/rsna-knee-288px-12-slice-cache"],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
