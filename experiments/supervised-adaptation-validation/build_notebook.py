import json
from pathlib import Path

root = Path(__file__).resolve().parent
base = (root.parent / "physical-paired-validation" / "physical_paired_validation.py").read_text(encoding="utf-8")
base = base.replace('if __name__=="__main__": main()', "")
extension = (root / "supervised_adaptation_validation.py").read_text(encoding="utf-8")
source = base + "\n\n" + extension
notebook = {
    "cells": [
        {"id": "adapt-intro", "cell_type": "markdown", "metadata": {}, "source": ["# RSNA Knee supervised adaptation\n", "Private leakage-safe validation; no submission.\n"]},
        {"id": "adapt-run", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "supervised-adaptation-validation.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
