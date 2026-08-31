import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "physical_preprocessing_audit.py").read_text(encoding="utf-8")
notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# RSNA Knee physical preprocessing audit\n",
            "Private, header-only preflight for a paired physical-space representation experiment. No model is fitted and no submission is created.\n",
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(root / "physical-preprocessing-audit.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
