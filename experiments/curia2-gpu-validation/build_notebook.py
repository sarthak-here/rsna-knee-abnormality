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

if patched != 1:
    raise RuntimeError(f"expected one Curia-2 accelerator patch, observed {patched}")

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
    "id": "sarthaksharma14/rsna-knee-curia2-gpu-validation",
    "title": "RSNA Knee Curia-2 GPU Private Validation",
    "code_file": OUTPUT.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
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
