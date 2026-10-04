import json
from pathlib import Path

root = Path(__file__).resolve().parent
base = (root.parent / "physical-paired-validation" / "physical_paired_validation.py").read_text(encoding="utf-8")
base = base.replace('if __name__=="__main__": main()', "")
extension = (root / "end_to_end_dinov2_validation.py").read_text(encoding="utf-8")
source = base + "\n\n" + extension
notebook = {
    "cells": [
        {"id": "e2e-intro", "cell_type": "markdown", "metadata": {}, "source": [
            "# RSNA Knee full end-to-end DINOv2 screen\n",
            "Private one-fold validation. Full encoder fine-tuning; no submission.\n",
        ]},
        {"id": "e2e-run", "cell_type": "code", "execution_count": None,
         "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "end-to-end-dinov2-validation.ipynb").write_text(
    json.dumps(notebook), encoding="utf-8")
