import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "upstream-v62.ipynb"
OUTPUT = ROOT / "curia2-gpu-validation.ipynb"
SOURCE_SHA256 = "cd00cfe1c13ef24aa740a620c8242ddc1c2c0c49ac57fe5e369a2bb70e5cd220"

if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("upstream public notebook identity changed")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

old = '''        os.environ.setdefault("PJRT_DEVICE", "TPU")
        os.environ.setdefault("XLA_USE_BF16", "1")
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        if device.type != "xla":
            raise RuntimeError(f"V62 Curia-2 experiment requires XLA TPU, observed {device}")
        audit["accelerator"] = "Kaggle TPU via PyTorch/XLA"
        audit["xla_device"] = str(device)
'''
new = '''        if not torch.cuda.is_available():
            raise RuntimeError("Curia-2 private replay requires CUDA")
        device = torch.device("cuda:0")
        if torch.cuda.device_count() < 2:
            raise RuntimeError(f"Curia-2 private replay requires two GPUs, observed {torch.cuda.device_count()}")
        audit["accelerator"] = "Kaggle 2xT4 CUDA private replay"
        audit["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
'''

patched = 0
for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", ""))
    if old in source:
        cell["source"] = source.replace(old, new)
        patched += 1

model_search_old = '''    candidates = []
    for config in Path("/kaggle/input").rglob("config.json"):
'''
model_search_new = '''    configs = list(Path("/kaggle/input").rglob("config.json"))
    try:
        import kagglehub
        downloaded = Path(kagglehub.model_download("olliebearman/curia-2/transformers/default"))
        import shutil
        staged = Path("/kaggle/working/curia2-pinned")
        staged.mkdir(parents=True, exist_ok=True)
        for name in (
            "model.safetensors",
            "config.json",
            "preprocessor_config.json",
            "curia_image_processor.py",
        ):
            shutil.copy2(downloaded / name, staged / name)
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/raidium/curia/resolve/main/LICENSE",
            staged / "LICENSE",
        )
        configs.extend(staged.rglob("config.json"))
        globals()["_V64_CURIA_MODEL_DELIVERY"] = "writable hash-pinned mirror plus official RAIL-M license"
    except Exception as exc:
        globals()["_V64_CURIA_MODEL_DOWNLOAD_ERROR"] = f"{type(exc).__name__}: {exc}"
    candidates = []
    for config in configs:
'''

model_search_patched = 0
for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", ""))
    if model_search_old in source:
        cell["source"] = source.replace(model_search_old, model_search_new)
        model_search_patched += 1

if patched != 1:
    raise RuntimeError(f"expected one Curia-2 accelerator patch, observed {patched}")
if model_search_patched != 1:
    raise RuntimeError(f"expected one Curia-2 delivery patch, observed {model_search_patched}")

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "id": "private-curia-cleanup",
    "metadata": {},
    "outputs": [],
    "source": '''# Private validation safety: retain reports, OOF, and checkpoints but remove
# every competition-ready CSV from Kaggle working output.
from pathlib import Path as _Path
_removed = []
for _path in _Path("/kaggle/working").rglob("submission*.csv"):
    _removed.append(str(_path))
    _path.unlink()
print("private validation cleanup removed", len(_removed), "submission CSV artifacts")
''',
})

notebook.setdefault("metadata", {})["curia2_gpu_replay"] = {
    "source": "prvsiyan/rsna-knee-read-the-report-then-the-knee version 62",
    "scientific_change": "accelerator-only TPU-to-2xT4 adaptation",
    "submission_capability": False,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

metadata = {
    "id": "sarthaksharma14/rsna-knee-curia-2-gpu-private-validation",
    "title": "RSNA Knee Curia-2 GPU Private Validation",
    "code_file": OUTPUT.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": [
        "marwanmath/resnet-50-radimagenet-marwan",
        "stevenleehans/rsna-knee-llm-report-labels",
        "lixin73/rsna-knee-llm-report-labels-sol56",
        "pilkwang/rsna-knee-llm-labels",
        "pilkwang/rsna-knee-weights",
        "royceplayz/rsna-medicalnet-resnet50-weights",
        "tonylica/rsna2026-models",
    ],
    "kernel_sources": [
        "gourabroy553/rsna-knee-hybrid-dino-radiomics",
        "blacklions/report-teacher-anatomy-aware-hierarchical-multimod",
    ],
    "model_sources": [
        "mrquadian/biomedclip/PyTorch/default/1",
        "metaresearch/dinov2/PyTorch/base/1",
        "metaresearch/dinov2/PyTorch/small/1",
        "timm/tf-efficientnet/PyTorch/tf-efficientnet-b3/1",
        "olliebearman/curia-2/Transformers/default/1",
    ],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
