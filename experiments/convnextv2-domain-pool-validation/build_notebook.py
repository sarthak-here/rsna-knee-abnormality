import json
from pathlib import Path


root = Path(__file__).resolve().parent
source = (root / "convnextv2_domain_pool_validation.py").read_text(encoding="utf-8")
public_cache = json.loads(
    (root.parent / "highres-pixel-cache" / "source-public-cache.ipynb").read_text(
        encoding="utf-8"))
preprocessing = """from __future__ import annotations
import os
os.environ['RSNA_IMG'] = '288'
os.environ['RSNA_CROP_MM'] = '140'
os.environ['RSNA_N_GROUP_MAX'] = '4'
os.environ['RSNA_WINDOW'] = '0.08,0.92'
"""
preprocessing += "\n\n".join(
    "".join(public_cache["cells"][index]["source"]).replace(
        "from __future__ import annotations\n", "")
    for index in range(10, 16)
    if public_cache["cells"][index]["cell_type"] == "code")
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "purpose",
            "metadata": {},
            "source": [
                "# RSNA Knee ConvNeXtV2 domain-pool screen\n",
                "Private leakage-safe validation of an attributed public RSNA-trained backbone. No submission.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "preprocessing",
            "metadata": {},
            "outputs": [],
            "source": preprocessing.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "run",
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "convnextv2-domain-pool-validation.ipynb").write_text(
    json.dumps(notebook), encoding="utf-8")
