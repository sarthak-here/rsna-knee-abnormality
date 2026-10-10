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
metadata.update({
    "id": "sarthaksharma14/rsna-knee-curia-2-private-reproduction",
    "title": "RSNA Knee Curia-2 Private Reproduction",
    "code_file": OUTPUT.name,
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
})
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
