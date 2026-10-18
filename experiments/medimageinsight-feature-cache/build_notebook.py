"""Build the private frozen MedImageInsight physical feature-cache notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = r'''from pathlib import Path
import hashlib, json, sys, time, types
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

H5_SHA = "490e5ccc07143699d05a9eb3ef9583e08795f1c2bacc6d9cf2bd0a9268158526"
STUDIES_SHA = "efc63afee30345f94a916f98ba6b30f266fccfe2976b8475fa86653b2c510513"
MODEL_SHA = "5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414"
CONFIG_SHA = "fb1ef53988b0128b4ef5ad829862cc80347cc8f4b009cdc70d8534f797c2851c"
WRAPPER_SHA = "0cd99b99aaa1b4f762ece21b8a53a57b95b41f86713a45e13bf65fa634252c0b"
EXPECTED_SHAPE = (4407, 6, 4, 1024)
EXPECTED_OCCUPIED = 23089
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
    raise RuntimeError("GPU required for MedImageInsight feature extraction")

started = time.time()
h5_path = unique_input("train_288x12.h5")
studies_path = unique_input("highres_cache_studies.csv")
model_path = unique_input("medimageinsigt-v1.0.0.pt")
config_path = unique_input("config.yaml")
wrapper_path = unique_input("medimageinsightmodel.py")
if h5_path.stat().st_size != 17051786047 or sha256(h5_path) != H5_SHA:
    raise RuntimeError("physical cache identity drift")
if sha256(studies_path) != STUDIES_SHA:
    raise RuntimeError("physical study manifest drift")
if model_path.stat().st_size != 2464060700 or sha256(model_path) != MODEL_SHA:
    raise RuntimeError("MedImageInsight checkpoint identity drift")
if sha256(config_path) != CONFIG_SHA or sha256(wrapper_path) != WRAPPER_SHA:
    raise RuntimeError("MedImageInsight source/config identity drift")

ids = pd.read_csv(studies_path, dtype=str).StudyInstanceUID.astype(str).to_numpy()
if len(ids) != 4407 or len(set(ids)) != 4407:
    raise RuntimeError("study identity/count drift")

model_root = wrapper_path.parent
sys.path.insert(0, str(model_root))

# The public source imports optional training-analysis packages at module import
# time although frozen image encoding does not use them. Supply minimal inert
# interfaces only when Kaggle's image lacks those packages; model construction
# remains standard-parameterized and the checkpoint is loaded strictly by the
# published implementation.
try:
    import mup  # noqa: F401
except ImportError:
    mup_module = types.ModuleType("mup")
    mup_init = types.ModuleType("mup.init")
    mup_init.trunc_normal_ = torch.nn.init.trunc_normal_
    mup_init.normal_ = torch.nn.init.normal_
    class MuReadout(torch.nn.Linear):
        def __init__(self, *args, readout_zero_init=False, **kwargs):
            super().__init__(*args, **kwargs)
            if readout_zero_init:
                torch.nn.init.zeros_(self.weight)
                if self.bias is not None:
                    torch.nn.init.zeros_(self.bias)
    mup_module.MuReadout = MuReadout
    mup_module.set_base_shapes = lambda *args, **kwargs: None
    mup_module.init = mup_init
    sys.modules["mup"] = mup_module
    sys.modules["mup.init"] = mup_init
try:
    import fvcore.nn  # noqa: F401
except ImportError:
    fvcore_module = types.ModuleType("fvcore")
    fvcore_nn = types.ModuleType("fvcore.nn")
    def _unused_fvcore(*args, **kwargs):
        raise RuntimeError("optional fvcore analysis path must not be called")
    fvcore_nn.FlopCountAnalysis = _unused_fvcore
    fvcore_nn.flop_count_table = _unused_fvcore
    fvcore_nn.flop_count_str = _unused_fvcore
    fvcore_module.nn = fvcore_nn
    sys.modules["fvcore"] = fvcore_module
    sys.modules["fvcore.nn"] = fvcore_nn
try:
    import yacs.config  # noqa: F401
except ImportError:
    yacs_module = types.ModuleType("yacs")
    yacs_config = types.ModuleType("yacs.config")
    yacs_config.CfgNode = dict
    yacs_module.config = yacs_config
    sys.modules["yacs"] = yacs_module
    sys.modules["yacs.config"] = yacs_config
from medimageinsightmodel import MedImageInsight

wrapper = MedImageInsight(
    model_dir=str(config_path.parent),
    vision_model_name=model_path.name,
    language_model_name="language_model.pth",
)
wrapper.load_model()
encoder = wrapper.model.eval()
encoder.image_encoder.enable_checkpoint = False
encoder.lang_encoder = None
wrapper.tokenize = None
for parameter in encoder.parameters():
    parameter.requires_grad_(False)
if getattr(wrapper.opt, "get", lambda *_: None)("UNICL_MODEL") is None:
    raise RuntimeError("MedImageInsight configuration load drift")

features = np.zeros(EXPECTED_SHAPE, np.float16)
all_masks = np.zeros((4407, 6), np.uint8)

@torch.inference_mode()
def encode(array):
    tensors = [wrapper.preprocess(Image.fromarray(np.moveaxis(image, 0, -1), mode="RGB"))
               for image in array]
    tensor = torch.stack(tensors).to(DEVICE, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16):
        embedding = encoder.encode_image(tensor).float()
        embedding = F.normalize(embedding, dim=1)
    if embedding.ndim != 2 or embedding.shape[1] != 1024:
        raise RuntimeError(f"embedding shape drift: {tuple(embedding.shape)}")
    return embedding.cpu().numpy().astype(np.float16)

with h5py.File(h5_path, "r") as handle:
    pixels, masks = handle["pixels"], handle["masks"]
    if pixels.shape != (4407, 6, 12, 288, 288) or masks.shape != (4407, 6):
        raise RuntimeError("physical cache tensor drift")
    for start in range(0, 4407, 4):
        stop = min(start + 4, 4407)
        block = pixels[start:stop]
        block_masks = np.asarray(masks[start:stop]).astype(np.uint8)
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
            for offset in range(0, len(array), 8):
                encoded.append(encode(array[offset:offset + 8]))
            for vector, (row, slot, window) in zip(np.concatenate(encoded), assignments):
                features[row, slot, window] = vector
        if start % 100 == 0:
            print(f"MedImageInsight encoded {start}/4407", flush=True)

if features.shape != EXPECTED_SHAPE or not np.isfinite(features).all():
    raise RuntimeError("feature output drift")
if all_masks.shape != (4407, 6) or not np.isin(all_masks, (0, 1)).all():
    raise RuntimeError("mask output drift")
if int(all_masks.sum()) != EXPECTED_OCCUPIED or (all_masks.sum(1) == 0).any():
    raise RuntimeError("occupied-slot count drift")
occupied_norms = np.linalg.norm(features.astype(np.float32), axis=-1)[all_masks.astype(bool)]
if not np.allclose(occupied_norms, 1.0, atol=0.003, rtol=0.003):
    raise RuntimeError("embedding normalization drift")
for row in range(4407):
    for slot in range(6):
        nonzero = bool(np.any(features[row, slot]))
        if nonzero != bool(all_masks[row, slot]):
            raise RuntimeError(f"feature/mask mismatch at {row},{slot}")

feature_path = OUT / "medimageinsight_physical_features.npz"
safe_ids = ids.astype("U")
np.savez_compressed(
    feature_path, ids=safe_ids, features=features, masks=all_masks,
    model_sha256=MODEL_SHA, config_sha256=CONFIG_SHA,
    wrapper_sha256=WRAPPER_SHA, h5_sha256=H5_SHA, studies_sha256=STUDIES_SHA,
)
receipt = {
    "status": "COMPLETE",
    "contract": "frozen MedImageInsight physical feature cache; no labels; no submission",
    "model_source": "thelionai/medimageinsight/PyTorch/default/1",
    "study_count": 4407, "unique_studies": int(len(set(ids))),
    "ids_dtype": str(safe_ids.dtype), "pickle_free": True,
    "feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
    "occupied_slots": int(all_masks.sum()), "finite": bool(np.isfinite(features).all()),
    "unit_normalized": True,
    "model_sha256": MODEL_SHA, "config_sha256": CONFIG_SHA,
    "wrapper_sha256": WRAPPER_SHA, "h5_sha256": H5_SHA,
    "studies_sha256": STUDIES_SHA, "output_sha256": sha256(feature_path),
    "runtime_seconds": time.time() - started, "submission_created": False,
}
(OUT / "medimageinsight_feature_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
for csv_path in OUT.glob("*.csv"):
    csv_path.unlink()
if list(OUT.glob("*.csv")):
    raise RuntimeError("forbidden CSV artifact")
print(json.dumps(receipt, indent=2), flush=True)
'''

compile(SOURCE, "medimageinsight-feature-cache", "exec")
notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# Frozen MedImageInsight physical feature cache\n",
                    "MRI-aware medical representation; no labels and no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run", "metadata": {},
         "outputs": [], "source": SOURCE.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(ROOT / "medimageinsight-feature-cache.ipynb").write_text(json.dumps(notebook, indent=1))
metadata = {
    "id": "sarthaksharma14/rsna-knee-medimageinsight-feature-cache",
    "title": "RSNA Knee MedImageInsight Feature Cache",
    "code_file": "medimageinsight-feature-cache.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "competition_sources": [], "dataset_sources": [],
    "kernel_sources": ["sarthaksharma14/rsna-knee-288px-12-slice-cache"],
    "model_sources": ["thelionai/medimageinsight/PyTorch/default/1"],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
