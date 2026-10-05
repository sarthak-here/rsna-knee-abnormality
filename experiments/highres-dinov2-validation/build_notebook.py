import json
from pathlib import Path


root = Path(__file__).resolve().parent
source = (root / "highres_dinov2_validation.py").read_text(encoding="utf-8")
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "purpose",
            "metadata": {},
            "source": [
                "# RSNA Knee high-resolution DINOv2 screen\n",
                "Private leakage-safe validation on the authenticated 288px/12-slice cache. No submission.\n",
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
(root / "highres-dinov2-validation.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
