import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "highres-pixel-cache" / "source-public-cache.ipynb"
notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
for index, cell in enumerate(notebook["cells"]):
    cell.setdefault("id", f"cell-{index:02d}")

notebook["cells"][0]["source"] = [
    "# Public CoAtNet-2 gold reproduction\n",
    "Inference-only reconstruction of the documented Apache-2.0 t4 checkpoint contract.\n",
    "No fitting, selection, competition prediction, or submission artifact is possible.\n",
]
notebook["cells"][3]["source"] = [
    "import os\n",
    "os.environ['RSNA_IMG'] = '280'\n",
    "os.environ['RSNA_CROP_MM'] = '150'\n",
    "os.environ['RSNA_N_GROUP_MAX'] = '5'\n",
    "for k, v in os.environ.items():\n",
    "    if k.startswith('RSNA_'):\n",
    "        print(f'{k} = {v}')\n",
]

runner = r'''
import hashlib
import json
import time

EXPECTED_MODEL_SHA256 = "5e9f9d3277924e2c0d86efc448447ef0f0c83dcf709078233403a50a5a9f2aca"
EXPECTED_CONFIG = {
    "version": "v4", "size": 280, "crop_mm": 150.0, "n_slices": 15,
    "n_slots": 6, "window": 3, "windows_per_slot": 2,
    "arch": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
}

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def unique_input(name):
    hits = list(Path("/kaggle/input").rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(hits)}: {hits[:8]}")
    return hits[0]

model_path = unique_input("coatnet2_fold0.pt")
config_path = unique_input("run_config.json")
if file_sha256(model_path) != EXPECTED_MODEL_SHA256:
    raise RuntimeError("public CoAtNet-2 checkpoint hash mismatch")
run_config = json.loads(config_path.read_text())
for key, expected in EXPECTED_CONFIG.items():
    if run_config.get(key) != expected:
        raise RuntimeError(f"run config mismatch for {key}: {run_config.get(key)!r} != {expected!r}")

root = find_root()
train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
gold = train[TARGETS].notna().all(axis=1)
gold_ids = train.loc[gold, "StudyInstanceUID"].tolist()
if len(gold_ids) != 58 or len(set(gold_ids)) != 58:
    raise RuntimeError(f"expected 58 unique gold studies, found {len(gold_ids)}")

series = pd.read_csv(root / "train_series.csv")
plane_map = dict(zip(series["SeriesInstanceUID"], series["Anatomical_Plane"]))
headers = annotate(walk(root, "train_series"))
headers["plane"] = headers["SeriesInstanceUID"].map(plane_map)
laterality = {study: resolve_laterality(group)
              for study, group in headers.groupby("StudyInstanceUID")}
all_slots = pick_slots(headers, plane_map)
missing = [uid for uid in gold_ids if uid not in all_slots]
if missing:
    raise RuntimeError(f"gold studies missing from slot map: {missing[:5]}")
subset = {uid: all_slots[uid] for uid in gold_ids}
t0 = time.time()
studies, cache, slot_mask = build_cache(subset, plane_map, laterality, "coatnet2-gold", 5)
if studies != sorted(gold_ids):
    raise RuntimeError("decoded study order drift")
if cache.shape != (58, 6, 15, 280, 280) or slot_mask.shape != (58, 6):
    raise RuntimeError(f"cache shape drift: {cache.shape}, {slot_mask.shape}")
if not np.isfinite(cache).all() or not np.isin(slot_mask, [0, 1]).all():
    raise RuntimeError("non-finite pixels or non-binary slot mask")
if np.any((slot_mask > 0) & ~np.any(cache != 0, axis=(2, 3, 4))):
    raise RuntimeError("occupied all-zero slot")

import timm

class ReleasedCoAtNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.empty(12, 256))
        self.out_weight = nn.Parameter(torch.empty(12, 256))
        self.out_bias = nn.Parameter(torch.empty(12))
        self.register_buffer("mean", torch.empty(1, 3, 1, 1))
        self.register_buffer("std", torch.empty(1, 3, 1, 1))
        self.backbone = timm.create_model(
            EXPECTED_CONFIG["arch"], pretrained=False, num_classes=0,
            global_pool="avg", drop_path_rate=0.1)
        if self.backbone.num_features != 1024:
            raise RuntimeError(f"unexpected backbone dimension {self.backbone.num_features}")
        self.proj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU())
        self.slot_emb = nn.Embedding(6, 256)

    def forward(self, pixels, slots):
        # pixels [B,S,G,3,H,W], where each G item is one physical adjacent triplet.
        b, s, g = pixels.shape[:3]
        valid = slots[:, :, None].expand(b, s, g).reshape(-1).bool()
        flat = pixels.reshape(b * s * g, 3, pixels.shape[-2], pixels.shape[-1])
        feat = self.out_weight.new_zeros((b * s * g, 1024))
        if valid.any():
            x = flat[valid].float().div_(255.0)
            x = (x - self.mean) / self.std
            feat[valid] = self.backbone(x.contiguous(memory_format=torch.channels_last))
        h = self.proj(feat).view(b, s, g, 256)
        slot_index = torch.arange(s, device=h.device).view(1, s, 1)
        h = h + self.slot_emb(slot_index)
        h = h.reshape(b, s * g, 256)
        token_valid = slots[:, :, None].expand(b, s, g).reshape(b, s * g).bool()
        att = torch.einsum("bth,oh->bot", h, self.query) / (256 ** 0.5)
        att = att.masked_fill(~token_valid[:, None], -1e4).softmax(-1)
        context = torch.einsum("bot,bth->boh", att, h)
        return (context * self.out_weight[None]).sum(-1) + self.out_bias

state = torch.load(model_path, map_location="cpu", weights_only=True)
model = ReleasedCoAtNet2()
missing_keys, unexpected_keys = model.load_state_dict(state, strict=True)
if missing_keys or unexpected_keys:
    raise RuntimeError(f"strict state mismatch: {missing_keys}, {unexpected_keys}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("GPU required")
model = model.to(device).eval()

# Five anchor triplets per slot; no windows are formed across anchor boundaries.
bags = cache.reshape(58, 6, 5, 3, 280, 280)
mask_tensor = torch.from_numpy(slot_mask).to(device)

@torch.inference_mode()
def predict(flip=False):
    output = []
    for start in range(0, len(bags), 2):
        x = torch.from_numpy(bags[start:start + 2]).to(device)
        if flip:
            x = x.flip(-1)
        with torch.autocast("cuda", dtype=torch.float16):
            output.append(torch.sigmoid(model(x, mask_tensor[start:start + 2])).float().cpu().numpy())
    return np.concatenate(output)

plain = predict(False)
flipped = predict(True)
average = (plain + flipped) / 2
if not all(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
           for p in (plain, flipped, average)):
    raise RuntimeError("invalid probabilities")

truth_by_id = train.set_index("StudyInstanceUID").loc[studies, TARGETS].to_numpy(np.float32)
from sklearn.metrics import roc_auc_score
def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]))

plain_auc = macro_auc(truth_by_id, plain)
flip_auc = macro_auc(truth_by_id, flipped)
average_auc = macro_auc(truth_by_id, average)
np.savez_compressed(Path("/kaggle/working/coatnet2_gold_predictions.npz"),
                    ids=np.asarray(studies), truth=truth_by_id,
                    plain=plain, flipped=flipped, average=average)
report = {
    "status": "COMPLETE",
    "contract": "inference-only public checkpoint reconstruction; no fitting or submission",
    "source_model": "patimakornsrinamen/rsna-67knee-coatnet2/PyTorch/t4/1",
    "source_license": "Apache-2.0",
    "checkpoint_sha256": EXPECTED_MODEL_SHA256,
    "run_config_sha256": file_sha256(config_path),
    "documented_gold_auc": float(run_config["gold58_auc"]),
    "documented_tta_gold_auc": float(run_config["tta_gold58_auc"]),
    "gold_count": 58,
    "fit_count": 0,
    "cache_shape": list(cache.shape),
    "occupied_slots": int(slot_mask.sum()),
    "order_stats": dict(ORDER_STATS),
    "pixel_stats": dict(PIX_STATS),
    "plain_gold_auc": plain_auc,
    "flipped_gold_auc": flip_auc,
    "predeclared_average_gold_auc": average_auc,
    "reproduction_gate": {"minimum_average_auc": 0.89, "passed": average_auc >= 0.89},
    "elapsed_seconds": time.time() - t0,
    "prediction_sha256": file_sha256("/kaggle/working/coatnet2_gold_predictions.npz"),
    "submission_created": False,
}
Path("/kaggle/working/coatnet2_gold_reproduction_report.json").write_text(
    json.dumps(report, indent=2) + "\n")
for forbidden in ("submission.csv", "submission.parquet"):
    if Path("/kaggle/working", forbidden).exists():
        raise RuntimeError(f"forbidden submission artifact exists: {forbidden}")
print(json.dumps(report, indent=2), flush=True)
'''

notebook["cells"][25]["source"] = runner.splitlines(keepends=True)
notebook["cells"][25]["outputs"] = []
notebook["cells"][25]["execution_count"] = None
(ROOT / "coatnet2-public-gold-reproduction.ipynb").write_text(
    json.dumps(notebook), encoding="utf-8")

metadata = {
    "id": "sarthaksharma14/rsna-knee-coatnet2-public-gold-reproduction",
    "title": "RSNA Knee CoAtNet2 Public Gold Reproduction",
    "code_file": "coatnet2-public-gold-reproduction.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": [],
    "kernel_sources": [],
    "model_sources": ["patimakornsrinamen/rsna-67knee-coatnet2/PyTorch/t4/1"],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
