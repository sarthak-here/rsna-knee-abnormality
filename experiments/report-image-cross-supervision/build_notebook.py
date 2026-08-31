import json
from pathlib import Path

root = Path(__file__).resolve().parent
base_path = root.parent / "physical-paired-validation" / "physical_paired_validation.py"
base = base_path.read_text(encoding="utf-8")
base = base.replace('if __name__=="__main__": main()', "")
extension = (root / "report_image_validation.py").read_text(encoding="utf-8")
source = base + "\n\n" + extension
notebook = {
    "cells": [
        {"id": "cross-supervision-intro", "cell_type": "markdown", "metadata": {}, "source": [
            "# RSNA Knee report-image cross-supervision\n",
            "Private leakage-safe validation. Reports are training-only; evaluation is image-only; all 58 gold studies are evaluation-only. No submission is created.\n",
        ]},
        {"id": "cross-supervision-run", "cell_type": "code", "execution_count": None,
         "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(root / "report-image-cross-supervision.ipynb").write_text(
    json.dumps(notebook), encoding="utf-8"
)
