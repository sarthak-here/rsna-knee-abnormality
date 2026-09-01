import json
from pathlib import Path

root = Path(__file__).resolve().parent
base = (root.parent / "physical-paired-validation" / "physical_paired_validation.py").read_text(encoding="utf-8")
base = base.replace('if __name__=="__main__": main()', "")
extension = (root / "segmentation_roi_validation.py").read_text(encoding="utf-8")
source = base + "\n\n" + extension
notebook = {
    "cells": [
        {
            "id": "segmentation-roi-intro",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RSNA Knee segmentation-guided ROI validation\n",
                "Private attributed KneeXNet anatomy-crop comparison. All 58 gold studies remain final evaluation-only, and this notebook cannot submit.\n",
            ],
        },
        {
            "id": "segmentation-roi-run",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "segmentation-roi-validation.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
