"""Build the private MedicalNet validation notebook from the reviewed script."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
source = (ROOT / "medicalnet_validation.py").read_text(encoding="utf-8")
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Private MedicalNet 3D held-out validation\n",
                "\n",
                "This experiment is validation-only. It does not produce a competition submission. "
                "The 58 organizer-labelled studies remain completely excluded from training.\n",
                "\n",
                "Attribution: Tencent MedicalNet / Med3D (MIT); public Kaggle weights from "
                "royceplayz; public weak labels from Steven Lee Hans and credited contributors.\n",
            ],
        },
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "medicalnet-3d-validation.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
