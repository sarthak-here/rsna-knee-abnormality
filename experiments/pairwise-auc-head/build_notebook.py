"""Build the private pairwise-AUC validation notebook from the pinned cache screen.

The generated notebook is self-contained. Reusing the already-audited loading,
identity, anchor, bootstrap, and no-CSV code keeps those contracts byte-for-byte
aligned while replacing only the declared architecture and objective.
"""

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_BUILDER = ROOT.parent / "crossplane-token-residual" / "build_notebook.py"
base = runpy.run_path(str(BASE_BUILDER))["SOURCE"]


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one replacement anchor, found {text.count(old)}: {old[:80]!r}")
    return text.replace(old, new, 1)


source = replace_once(base, "import gc, hashlib, json, math, random\n", "import gc, hashlib, json, math, random, sys\n")
source = replace_once(source, "EPOCHS = 24", "EPOCHS = 20")
source = replace_once(
    source,
    'ANCHOR_SHA = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"\n',
    'ANCHOR_SHA = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"\n'
    'KNEE_SHA = "314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1"\n',
)

class_start = source.index("class CrossPlaneTargetQuery")
device_start = source.index('if DEVICE.type != "cuda":', class_start)
objective = r'''PAIR_THRESHOLD = .10
PAIR_WEIGHT = .85
BCE_WEIGHT = .15

def weighted_soft_pairwise_auc(logits, targets, threshold=PAIR_THRESHOLD):
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must have identical [batch,target] shape")
    losses = []
    for target_index in range(logits.shape[1]):
        y = targets[:, target_index].float()
        z = logits[:, target_index].float()
        target_delta = y[:, None] - y[None, :]
        keep = target_delta >= threshold
        if not bool(keep.any()):
            continue
        logit_delta = z[:, None] - z[None, :]
        weights = target_delta[keep]
        losses.append((F.softplus(-logit_delta[keep]) * weights).sum() / weights.sum())
    if not losses:
        raise RuntimeError("batch contains no target pairs above the fixed threshold")
    return torch.stack(losses).mean()

def pairwise_bce_objective(logits, targets):
    ranking = weighted_soft_pairwise_auc(logits, targets)
    calibration = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    return PAIR_WEIGHT * ranking + BCE_WEIGHT * calibration

'''
source = source[:class_start] + objective + source[device_start:]

source = replace_once(
    source,
    'if sha256(anchor_path) != ANCHOR_SHA:\n    raise RuntimeError("anchor cache SHA drift")\n',
    'if sha256(anchor_path) != ANCHOR_SHA:\n    raise RuntimeError("anchor cache SHA drift")\n'
    'knee_path = unique_input("knee.py")\n'
    'if sha256(knee_path) != KNEE_SHA:\n    raise RuntimeError("public head source SHA drift")\n'
    'sys.path.insert(0, str(knee_path.parent))\n'
    'from knee import KneeNet\n',
)
source = replace_once(
    source,
    "    model = CrossPlaneTargetQuery().to(DEVICE)\n",
    '    model = KneeNet("convnext_tiny", pretrained=False)\n'
    '    del model.enc\n'
    '    model = model.to(DEVICE)\n',
)
source = replace_once(
    source,
    "            output.append(torch.sigmoid(model(x, m, p)).float().cpu().numpy())\n",
    "            output.append(torch.sigmoid(model.head(x, m.bool(), p)).float().cpu().numpy())\n",
)
source = replace_once(source, "                logits = model(x, mask, pos)\n", "                logits = model.head(x, mask.bool(), pos)\n")
source = replace_once(
    source,
    "                loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())\n",
    "                loss = pairwise_bce_objective(logits, y)\n",
)
source = replace_once(source, 'ckpt = OUT / f"crossplane_seed_{seed}.pt"', 'ckpt = OUT / f"pairwise_auc_seed_{seed}.pt"')
source = replace_once(
    source,
    '"contract": "fixed cross-plane target-query reader; 10% rank residual; no submission",',
    '"contract": "exact public attention head; 85% pairwise-AUC plus 15% BCE; fixed 10% rank blend; no submission",',
)
source = replace_once(
    source,
    '"feature_sha256": FEATURE_SHA,\n',
    '"feature_sha256": FEATURE_SHA,\n    "source_knee_sha256": KNEE_SHA,\n'
    '    "pair_threshold": PAIR_THRESHOLD,\n    "pair_weight": PAIR_WEIGHT,\n    "bce_weight": BCE_WEIGHT,\n',
)
source = replace_once(
    source,
    'np.savez_compressed(OUT / "crossplane_gold_predictions.npz",',
    'np.savez_compressed(OUT / "pairwise_auc_gold_predictions.npz",',
)
source = replace_once(
    source,
    '(OUT / "crossplane_token_residual_report.json").write_text(json.dumps(report, indent=2) + "\\n")',
    '(OUT / "pairwise_auc_head_report.json").write_text(json.dumps(report, indent=2) + "\\n")',
)

for forbidden in ("CrossPlaneTargetQuery", "crossplane_seed_", "crossplane_gold_predictions", "crossplane_token_residual_report"):
    if forbidden in source:
        raise RuntimeError(f"stale cross-plane symbol remains: {forbidden}")

notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# Pairwise-AUC ConvNeXt head screen\n",
                    "Fixed five-seed private ranking validation; organizer gold is evaluation-only; no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run", "metadata": {},
         "outputs": [], "source": source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "pairwise-auc-head.ipynb").write_text(json.dumps(notebook, indent=1))

metadata = {
    "id": "sarthaksharma14/rsna-knee-pairwise-auc-head",
    "title": "RSNA Knee Pairwise AUC Head",
    "code_file": "pairwise-auc-head.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader",
                        "stevenleehans/rsna-knee-llm-report-labels"],
    "kernel_sources": ["sarthaksharma14/rsna-knee-convnext-reader-feature-cache",
                       "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit",
                       "sarthaksharma14/rsna-knee-multireader-nested-blend-screen"],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
