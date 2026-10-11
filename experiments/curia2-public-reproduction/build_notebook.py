import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = ROOT.parent / "curia2-gpu-validation"
VALIDATION_BUILDER = VALIDATION_ROOT / "build_notebook.py"
VALIDATION_NOTEBOOK = VALIDATION_ROOT / "curia2-gpu-validation.ipynb"
OUTPUT = ROOT / "curia2-public-reproduction.ipynb"

# Rebuild the hash-pinned private-validation source first so this inference
# reproduction cannot drift from the validated scientific contract.
runpy.run_path(str(VALIDATION_BUILDER), run_name="__main__")
notebook = json.loads(VALIDATION_NOTEBOOK.read_text(encoding="utf-8"))

# Kaggle code submissions must run with internet disabled. Replace only the
# validated delivery helper: discover the exact public Curia assets from the
# attached private offline mirror, verify the pinned model hash, then stage the
# same model/config/preprocessor/source/license files into writable storage.
old_delivery = '''def _v64_curia_configs():
    configs = list(Path("/kaggle/input").rglob("config.json"))
    import kagglehub, shutil, urllib.request
    root = Path(kagglehub.model_download("olliebearman/curia-2/transformers/default"))
    staged = Path("/kaggle/working/curia2-pinned"); staged.mkdir(parents=True, exist_ok=True); [shutil.copy2(root / name, staged / name) for name in ("model.safetensors", "config.json", "preprocessor_config.json", "curia_image_processor.py")]
    if True:
        urllib.request.urlretrieve("https://huggingface.co/raidium/curia/resolve/main/LICENSE", staged / "LICENSE")
    configs.extend(staged.rglob("config.json"))
    globals()["_V64_CURIA_MODEL_DELIVERY"] = "writable hash-pinned mirror plus official RAIL-M license"
    return configs
'''
new_delivery = '''def _v64_curia_configs():
    configs = list(Path("/kaggle/input").rglob("config.json"))
    import shutil
    expected_model = "403a02e27531d2858ecd1e9b1ec2d5ea7bfa909f10ff0d9e8416090a6a8c96ef"
    mirrors = [
        path for path in Path("/kaggle/input").rglob("model.safetensors")
        if _v64_sha256(path) == expected_model
    ]
    if len(mirrors) != 1:
        raise RuntimeError(f"expected one exact offline Curia-2 model mirror, found {len(mirrors)}")
    root = mirrors[0].parent
    staged = Path("/kaggle/working/curia2-pinned")
    staged.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json", "preprocessor_config.json", "curia_image_processor.py", "LICENSE"):
        shutil.copy2(root / name, staged / name)
    configs.extend(staged.rglob("config.json"))
    globals()["_V64_CURIA_MODEL_DELIVERY"] = "offline hash-pinned mirror plus official RAIL-M license"
    return configs
'''
delivery_replacements = 0
for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = cell.get("source", "")
    if old_delivery in source:
        cell["source"] = source.replace(old_delivery, new_delivery)
        delivery_replacements += 1
if delivery_replacements != 1:
    raise RuntimeError(f"expected exactly one Curia delivery helper, replaced {delivery_replacements}")

before = len(notebook["cells"])
notebook["cells"] = [
    cell for cell in notebook["cells"]
    if cell.get("id") != "private-curia-cleanup"
]
if len(notebook["cells"]) != before - 1:
    raise RuntimeError("expected exactly one private cleanup cell")

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "id": "curia-public-receipt",
    "metadata": {},
    "outputs": [],
    "source": '''# Strict inference receipt: hidden reruns must fail closed.
import hashlib as _hashlib
import json as _json
from pathlib import Path as _Path
import numpy as _np
import pandas as _pd

_primary = _Path("/kaggle/working/submission.csv")
_audit_path = _Path("/kaggle/working/curia_v64_audit.json")
_receipt_path = _Path("/kaggle/working/curia2_reproduction_receipt.json")
if not _primary.is_file() or not _audit_path.is_file():
    raise FileNotFoundError("Curia candidate or audit artifact is absent")
_audit = _json.loads(_audit_path.read_text())
if _audit.get("status") != "CANDIDATE_SELECTED":
    raise RuntimeError(f"Curia candidate was not selected: {_audit.get('status')}")
_oof = _audit.get("oof", {})
if not bool(_oof.get("selection_supported")):
    raise RuntimeError("nested OOF selection gate is false")
if abs(float(_oof.get("final_alpha", -1)) - 0.20) > 1e-12:
    raise RuntimeError("selected alpha drift")
if int(_audit.get("head_count", -1)) != 5 or int(_audit.get("test_head_count", -1)) != 5:
    raise RuntimeError("five-head completeness gate failed")
if int(_oof.get("positive_outer_folds", -1)) < 3:
    raise RuntimeError("outer-fold support gate failed")
if float(_oof.get("nested_blend_macro_auc", 0)) < float(_oof.get("v52_rad20_macro_auc", 1)) + 0.001:
    raise RuntimeError("nested improvement gate failed")
if float(_oof.get("final_descriptive_macro_auc", 0)) < float(_oof.get("v52_rad20_macro_auc", 1)) + 0.001:
    raise RuntimeError("final descriptive improvement gate failed")

_frame = _pd.read_csv(_primary, dtype={"StudyInstanceUID": str})
_test = _pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
_expected_columns = ["StudyInstanceUID", *TARGETS]
if _frame.columns.tolist() != _expected_columns:
    raise RuntimeError("official submission column/order drift")
_ids = _frame["StudyInstanceUID"].astype(str).tolist()
_expected_ids = _test["StudyInstanceUID"].astype(str).tolist()
if _ids != _expected_ids or len(_ids) != len(set(_ids)):
    raise RuntimeError("hidden test identity/order/uniqueness drift")
_values = _frame[TARGETS].to_numpy(_np.float64)
if _values.shape != (len(_expected_ids), len(TARGETS)):
    raise RuntimeError("submission probability shape drift")
if not _np.isfinite(_values).all() or _values.min() < 0 or _values.max() > 1:
    raise RuntimeError("submission probabilities are invalid")

def _sha256(path):
    digest = _hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

_submission_sha = _sha256(_primary)
if _submission_sha != _audit.get("selected_sha256"):
    raise RuntimeError("selected submission SHA-256 drift")
_receipt = {
    "status": "COMPLETE",
    "notebook_contract": "source-faithful Curia-2 V64 two-GPU reproduction",
    "submission_sha256": _submission_sha,
    "rows": len(_frame),
    "unique_ids": len(set(_ids)),
    "columns": _expected_columns,
    "probability_min": float(_values.min()),
    "probability_max": float(_values.max()),
    "final_alpha": float(_oof["final_alpha"]),
    "nested_blend_macro_auc": float(_oof["nested_blend_macro_auc"]),
    "final_descriptive_macro_auc": float(_oof["final_descriptive_macro_auc"]),
    "positive_outer_folds": int(_oof["positive_outer_folds"]),
    "head_count": int(_audit["head_count"]),
    "test_head_count": int(_audit["test_head_count"]),
    "encoder_sha256": _audit["encoder_sha256"],
    "encoder_config_sha256": _audit["encoder_config_sha256"],
    "encoder_license": _audit["encoder_license"],
}
_receipt_path.write_text(_json.dumps(_receipt, indent=2, sort_keys=True) + "\\n")
print(_json.dumps(_receipt, indent=2, sort_keys=True))
''',
})

notebook.setdefault("metadata", {})["curia2_public_reproduction"] = {
    "source_validation": "sarthaksharma14/rsna-knee-curia-2-gpu-private-validation version 6",
    "validated_alpha": 0.20,
    "scientific_change": "none; retain the validated candidate and add fail-closed receipt",
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

metadata = json.loads((VALIDATION_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))
metadata["dataset_sources"] = list(dict.fromkeys([\n    *metadata.get("dataset_sources", []),\n    "sarthaksharma14/rsna-knee-curia2-pinned-offline-assets",\n]))\nmetadata.update({
    "id": "sarthaksharma14/rsna-knee-curia-2-private-reproduction",
    "title": "RSNA Knee Curia-2 Private Reproduction",
    "code_file": OUTPUT.name,
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
})
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
