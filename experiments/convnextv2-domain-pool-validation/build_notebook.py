import json
from pathlib import Path


root = Path(__file__).resolve().parent
source = (root / "convnextv2_domain_pool_validation.py").read_text(encoding="utf-8")
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
