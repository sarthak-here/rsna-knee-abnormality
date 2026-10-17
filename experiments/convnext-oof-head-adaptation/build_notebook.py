import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "convnext-reader-oof-head-validation" / "convnext-reader-oof-head-validation.ipynb"
BASE_SHA256 = "3c6004445a09aece15a4c15ef6b04b8dce5992ec4bc205b770138b6b6c32d900"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if sha256(BASE) != BASE_SHA256:
    raise RuntimeError("base validation notebook identity drift")
notebook = json.loads(BASE.read_text())
source = "".join(notebook["cells"][1]["source"])


def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"expected one replacement site, found {source.count(old)}: {old[:80]}")
    source = source.replace(old, new)


replace_once(
    'def new_head(seed):\n    seed_all(seed)\n    model = KneeNet("convnext_tiny", pretrained=False)\n    del model.enc\n    return model.to(DEVICE)\n',
    'def new_head(seed):\n    seed_all(seed)\n    model = KneeNet("convnext_tiny", pretrained=False)\n    model.load_state_dict(public_state, strict=True)\n    del model.enc\n    return model.to(DEVICE)\n',
)
replace_once(
    'def train_fixed(labels, seed, epochs=20):\n    model = new_head(seed)\n    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)',
    'def train_fixed(labels, seed, epochs=5):\n    model = new_head(seed)\n    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3)',
)
replace_once(
    '# Independently validate the original public fold-0 reader head on untouched gold.\ncheckpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)\noriginal = KneeNet(checkpoint["args"]["backbone"].split(".")[0], pretrained=False)\noriginal.load_state_dict(checkpoint["model"], strict=True)',
    '# Independently validate and freeze the original public fold-0 reader head.\ncheckpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)\npublic_state = checkpoint["model"]\noriginal = KneeNet(checkpoint["args"]["backbone"].split(".")[0], pretrained=False)\noriginal.load_state_dict(public_state, strict=True)',
)
replace_once('"epochs": 20,', '"epochs": 5,')
replace_once(
    '"contract": "private paired ConvNeXt-reader head target audit; no submission",',
    '"contract": "private paired adaptation of the public ConvNeXt head; no submission",\n    "base_notebook_sha256": "' + BASE_SHA256 + '",\n    "adaptation": {"epochs": 5, "learning_rate": 2e-5, "initialization": "exact public fold-0 head"},',
)
replace_once(
    '"checkpoint_selection": "none; fixed 20 epochs",',
    '"checkpoint_selection": "none; fixed five adaptation epochs",',
)
replace_once(
    '"gate": {"delta_at_least_0_005": candidate_auc - control_auc >= .005,\n             "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,\n             "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= .95,\n             "bootstrap_q025_positive": bootstrap["q025"] > 0,\n             "candidate_absolute_at_least_0_90": candidate_auc >= .90},',
    '"gate": {"paired_delta_at_least_0_002": candidate_auc - control_auc >= .002,\n             "candidate_minus_public_at_least_0_002": candidate_auc - original_gold_auc >= .002,\n             "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,\n             "bootstrap_p_at_least_0_90": bootstrap["p_improve"] >= .90,\n             "bootstrap_q025_above_minus_0_001": bootstrap["q025"] > -.001},',
)
replace_once(
    'Path("/kaggle/working/convnext_reader_oof_head_report.json").write_text(\n    json.dumps(report, indent=2) + "\\n")',
    'out = Path("/kaggle/working")\nPath(out / "convnext_oof_head_adaptation_report.json").write_text(\n    json.dumps(report, indent=2) + "\\n")\nnp.savez_compressed(out / "convnext_oof_head_adaptation_predictions.npz",\n    ids=ids[gold_idx].astype("U"), truth=gold_truth.astype(np.uint8),\n    original=original_gold_prediction.astype(np.float32),\n    control=np.asarray(control_gold, np.float32),\n    candidate=np.asarray(candidate_gold, np.float32),\n    seeds=np.asarray(SEEDS, np.int64))',
)
notebook["cells"][0]["source"] = [
    "# ConvNeXt OOF-corrected head adaptation\n",
    "Private paired adaptation from the exact public head; no submission.\n",
]
notebook["cells"][1]["source"] = source.splitlines(keepends=True)
(ROOT / "convnext-oof-head-adaptation.ipynb").write_text(json.dumps(notebook))

metadata = {
    "id": "sarthaksharma14/rsna-knee-convnext-oof-head-adaptation",
    "title": "RSNA Knee ConvNeXt OOF Head Adaptation",
    "code_file": "convnext-oof-head-adaptation.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": [
        "goodpjw2008/rsna-knee-2-5d-convnext-reader",
        "stevenleehans/rsna-knee-llm-report-labels",
    ],
    "kernel_sources": [
        "sarthaksharma14/rsna-knee-convnext-reader-feature-cache",
        "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit",
    ],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

