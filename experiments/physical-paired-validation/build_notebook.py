import json
from pathlib import Path
root=Path(__file__).resolve().parent
source=(root/"physical_paired_validation.py").read_text(encoding="utf-8")
nb={"cells":[{"cell_type":"markdown","metadata":{},"source":["# RSNA Knee paired physical preprocessing validation\n","Private leakage-safe comparison. The 58 gold studies are evaluation-only; this notebook cannot submit.\n"]},{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
(root/"physical-paired-validation.ipynb").write_text(json.dumps(nb),encoding="utf-8")
