"""Build the frozen-MedImageInsight target-attention residual screen."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_BUILDER = ROOT.parent / "biomedclip-residual-screen" / "build_notebook.py"
spec = importlib.util.spec_from_file_location("biomedclip_residual_builder", BASE_BUILDER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SOURCE = module.SOURCE

replacements = {
    'FEATURE_SHA = "40768bf7c57478732c1fb14c77510b4e650634222337364c109747d0d3e9c434"':
        'FEATURE_SHA = "2f8344d247185a7faca711057ef17ce4cd15df540dd3181501c2dfc85e18922a"',
    'MODEL_SHA = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"':
        'MODEL_SHA = "5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414"',
    'class TargetAttention(nn.Module):\n    def __init__(self, input_dim=512, dim=256, targets=12):':
        'class TargetAttention(nn.Module):\n    def __init__(self, input_dim=1024, dim=256, targets=12):',
    'unique_input("biomedclip_physical_features.npz")':
        'unique_input("medimageinsight_physical_features.npz")',
    '(4407, 6, 4, 512)': '(4407, 6, 4, 1024)',
    'f"biomedclip_head_seed_{seed}.pt"': 'f"medimageinsight_head_seed_{seed}.pt"',
    '"frozen BiomedCLIP target-attention reader; fixed 10% rank residual; no submission"':
        '"frozen MedImageInsight target-attention reader; fixed 10% rank residual; no submission"',
    '"biomedclip_residual_gold_predictions.npz"':
        '"medimageinsight_residual_gold_predictions.npz"',
    '"biomedclip_residual_report.json"': '"medimageinsight_residual_report.json"',
}
for old, new in replacements.items():
    count = SOURCE.count(old)
    if count != 1:
        raise RuntimeError(f"base harness drift for {old!r}: found {count}")
    SOURCE = SOURCE.replace(old, new)

compile(SOURCE, "medimageinsight-residual-screen", "exec")
notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# MedImageInsight target-attention residual screen\n",
                    "Fixed five-seed validation; organizer gold is evaluation-only; no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run", "metadata": {},
         "outputs": [], "source": SOURCE.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(ROOT / "medimageinsight-residual-screen.ipynb").write_text(json.dumps(notebook, indent=1))
metadata = {
    "id": "sarthaksharma14/rsna-knee-medimageinsight-residual-screen",
    "title": "RSNA Knee MedImageInsight Residual Screen",
    "code_file": "medimageinsight-residual-screen.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels"],
    "kernel_sources": ["sarthaksharma14/rsna-knee-medimageinsight-feature-cache",
                       "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit",
                       "sarthaksharma14/rsna-knee-multireader-nested-blend-screen"],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
