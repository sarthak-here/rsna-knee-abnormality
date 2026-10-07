import json
from pathlib import Path

root = Path(__file__).resolve().parent
base = (root.parent / "physical-paired-validation" /
        "physical_paired_validation.py").read_text(encoding="utf-8")
base = base.replace('if __name__=="__main__": main()', "")
shared = (root.parent / "convnext-oof-student-validation" /
          "oof_student_validation.py").read_text(encoding="utf-8")
call = "\n\nmain_oof_student()\n"
if call not in shared:
    raise RuntimeError("shared student source call marker drift")
shared = shared.replace(call, "\n")
extension = (root / "full_population_validation.py").read_text(encoding="utf-8")
source = base + "\n\n" + shared + "\n\n" + extension
notebook = {
    "cells": [
        {"id": "full-pop-intro", "cell_type": "markdown", "metadata": {},
         "source": ["# RSNA Knee full-population OOF target validation\n",
                    "Private fixed-epoch paired audit; no submission.\n"]},
        {"id": "full-pop-run", "cell_type": "code", "execution_count": None,
         "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "convnext-oof-full-population-validation.ipynb").write_text(
    json.dumps(notebook), encoding="utf-8")

