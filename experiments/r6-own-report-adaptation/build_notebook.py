"""Private bounded R6 own-report experiment; no original inference launcher."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / "multireader-nested-blend/build_notebook.py"
s = importlib.util.spec_from_file_location("original_assets_builder", BASE_PATH)
base = importlib.util.module_from_spec(s); s.loader.exec_module(base)


def build():
    if base.sha256(base.DEFAULT_R6) != base.R6_SHA256:
        raise ValueError("audited asset-source notebook changed")
    original = json.loads(base.DEFAULT_R6.read_text(encoding="utf-8"))
    assets_stage = base.patch_r6_frozen_mount(base.cell_source(original, 6))
    # This cell prepares pinned source/weights only. Original cell7 (inference
    # launcher), Raptor stages, hidden evaluation and submission are NOT included.
    compile(assets_stage, "pinned_assets", "exec")
    files = {name: (HERE / name).read_text(encoding="utf-8") for name in
             ("report_rules.py", "prepare_splits.py", "train.py", "splits.json")}
    for name in ("postprocess.py", "strict_reconstruct.py"):
        files[name] = (HERE.parent / "r6-fold0-warmstart" / name).read_text(encoding="utf-8")
    setup = '''import time, json, sys
from pathlib import Path
NB_T0 = time.time()
OWN = Path('/kaggle/working/own_report_adaptation')
OWN.mkdir(exist_ok=False)
'''
    setup += "FILES = " + repr(files) + "\n"
    setup += "for name, content in FILES.items():\n    (OWN/name).write_text(content)\n"
    launch = '''import hashlib, importlib.metadata, os, subprocess
import pandas as pd
roots = [p.parent for p in Path('/kaggle/input').rglob('train.csv')
         if (p.parent/'train_series.csv').exists() and (p.parent/'train_series').is_dir()]
if len(roots) != 1:
    raise RuntimeError('competition train mount is ambiguous: ' + str(roots))
root = roots[0]
split = json.loads((OWN/'splits.json').read_text())
if split['status'] != 'READY':
    raise RuntimeError('predeclared coverage gate failed')
ids = [r['id'] for k in ('fit','tune','confirmation') for r in split[k]]
evaluation = TMP/'own_report_eval'
evaluation.mkdir(exist_ok=False)
pd.DataFrame({'StudyInstanceUID': ids}).to_csv(evaluation/'test.csv', index=False)
series = pd.read_csv(root/'train_series.csv', dtype=str)
series = series[series.StudyInstanceUID.isin(ids)]
if set(series.StudyInstanceUID) != set(ids):
    raise RuntimeError('selected study series coverage mismatch')
series.to_csv(evaluation/'test_series.csv', index=False)
os.symlink(root/'train_series', evaluation/'test_series', target_is_directory=True)
assets = {'out': str(OWN/'results'), 'started_epoch': NB_T0, 'source': str(SRC), 'bundle': str(BUNDLE),
          'code': str(CODE), 'anatomy': str(CODE_SRC/'anatomy.pt'), 'train': str(root/'train.csv'),
          'eval_root': str(evaluation), 'cache': str(TMP/'own_report_features'), 'splits': str(OWN/'splits.json'),
          'split_sha256': hashlib.sha256((OWN/'splits.json').read_bytes()).hexdigest()}
(OWN/'assets.json').write_text(json.dumps(assets, indent=2))
environment = {p: importlib.metadata.version(p) for p in ('torch','torchvision','timm','numpy','pydicom')}
environment['source_notebook_sha256'] = '__SOURCE_SHA__'
(OWN/'environment.json').write_text(json.dumps(environment, indent=2))
try:
    subprocess.run([sys.executable, str(OWN/'train.py'), '--assets', str(OWN/'assets.json')], check=True,
                   timeout=max(1., NB_T0 + 10800 - time.time()))
except subprocess.TimeoutExpired:
    (OWN/'results').mkdir(exist_ok=True)
    (OWN/'results'/'budget_rejection.json').write_text(json.dumps({'stage': 'hard_wall_deadline', 'incomplete_candidate': True}))
    print('OWN_REPORT_HARD_BUDGET_STOP_NO_SUBMISSION', flush=True)
if list(Path('/kaggle/working').rglob('*.csv')):
    raise RuntimeError('forbidden CSV output in working directory')
print('PRIVATE_OWN_REPORT_NOTEBOOK_COMPLETE_NO_SUBMISSION', flush=True)
'''.replace("__SOURCE_SHA__", base.R6_SHA256)
    compile(setup, "setup", "exec"); compile(launch, "launch", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": original["metadata"], "cells": [
        base.markdown("# R6 own-report final-stage adaptation\nPrivate hypothesis screen; NOT a submission. Original R6/source/medical locator attribution: Heliosli and bundled upstream authors. Existing model-use terms and OAI/NDA acknowledgement retained. No new OrthoFoundation reader.\n"),
        base.code(setup), base.code(assets_stage), base.code(launch)]}
    path = HERE / "own-report-adaptation.ipynb"
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    metadata = {"id": "sarthaksharma14/rsna-knee-r6-own-report-adaptation", "title": "RSNA Knee R6 Own Report Adaptation",
                "code_file": path.name, "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
                "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
    (HERE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {"source_notebook_sha256": base.R6_SHA256,
               "files_sha256": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()},
               "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "submission_launcher_included": False}
    (HERE / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
