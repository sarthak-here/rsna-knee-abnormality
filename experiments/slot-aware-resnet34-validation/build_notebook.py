import json
from pathlib import Path

root = Path(__file__).resolve().parent
base = (root.parent / "physical-paired-validation" / "physical_paired_validation.py").read_text(encoding="utf-8")
base = base.replace('if __name__=="__main__": main()', "")
shared = (root.parent / "quality-efficientnet-validation" / "quality_efficientnet_validation.py").read_text(encoding="utf-8")
shared = shared.replace("\nmain_quality_efficientnet()\n", "\n")
extension = (root / "slot_aware_resnet34_validation.py").read_text(encoding="utf-8")
source = base + "\n\n" + shared + "\n\n" + extension
notebook = {
    "cells": [
        {"id": "slot-aware-resnet34-intro", "cell_type": "markdown", "metadata": {},
         "source": ["# RSNA Knee slot-aware ResNet34 screen\n",
                    "Private one-fold leakage-safe validation. No submission capability.\n"]},
        {"id": "slot-aware-resnet34-run", "cell_type": "code", "execution_count": None,
         "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "slot-aware-resnet34-validation.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
