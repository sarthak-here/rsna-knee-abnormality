"""Package the exact source for a private Kaggle research screen, never submission."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {name: (ROOT / name).read_text() for name in
         ("models.py", "search.py", "preflight.py", "fetch_preflight_assets.py")}
SOURCE = '''from pathlib import Path
import hashlib, importlib.metadata, json, subprocess, sys
ROOT = Path('/kaggle/working/transfer_optuna')
ROOT.mkdir(exist_ok=True)
'''
SOURCE += "FILES = " + repr(FILES) + "\n"
SOURCE += '''for name, content in FILES.items():
    (ROOT / name).write_text(content)
# Do not change torch/torchvision or their CUDA builds.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                'optuna==4.5.0', 'transformers==4.57.1', 'safetensors==0.6.2',
                'h5py==3.14.0'], check=True)
sys.path.insert(0, str(ROOT))
from models import sha256
subprocess.run([sys.executable, str(ROOT/'fetch_preflight_assets.py'),
                '--out', str(ROOT/'weights')], check=True)
assets_path = ROOT/'weights/assets.json'
subprocess.run([sys.executable, str(ROOT/'preflight.py'), '--assets', str(assets_path),
                '--out', str(ROOT/'preflight.json')], check=True)
assets = json.loads(assets_path.read_text())
def unique(name):
    hits = [p for p in Path('/kaggle/input').rglob(name) if p.is_file()]
    if len(hits) != 1:
        raise RuntimeError(f'expected unique {name}, found {hits}')
    return hits[0]
pins = {
    'h5': ('train_288x12.h5', '490e5ccc07143699d05a9eb3ef9583e08795f1c2bacc6d9cf2bd0a9268158526'),
    'studies': ('highres_cache_studies.csv', 'efc63afee30345f94a916f98ba6b30f266fccfe2976b8475fa86653b2c510513'),
    'train': ('train.csv', '8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33'),
    'labels': ('llm_labels_v4_blend.csv', 'LABEL_SHA_PLACEHOLDER'),
}
for key, (name, digest) in pins.items():
    assets[key] = {'path': str(unique(name)), 'sha256': digest}
assets['preflight'] = {'path': str(ROOT/'preflight.json'), 'sha256': sha256(ROOT/'preflight.json')}
# This is noncommercial private research only, not clearance for competition/prize use.
# Neither new encoder uses OAI weights. The OAI OrthoFoundation candidate is excluded.
assets['terms_reviewed'] = True
assets['terms_scope'] = 'private noncommercial no-submission research; VideoMAE CC-BY-NC-4.0'
assets_path.write_text(json.dumps(assets, indent=2) + '\\n')
environment = {name: importlib.metadata.version(name) for name in
               ('torch', 'torchvision', 'transformers', 'optuna', 'h5py', 'safetensors')}
(ROOT/'environment.json').write_text(json.dumps(environment, indent=2) + '\\n')
subprocess.run([sys.executable, str(ROOT/'search.py'), '--assets', str(assets_path),
                '--out', str(ROOT/'search-output')], check=True)
csvs = list(Path('/kaggle/working').rglob('*.csv'))
if csvs:
    raise RuntimeError(f'forbidden CSV output: {csvs}')
print('TRANSFER_OPTUNA_COMPLETE_NO_SUBMISSION', flush=True)
'''


def build(label_sha):
    source = SOURCE.replace("LABEL_SHA_PLACEHOLDER", label_sha)
    compile(source, "transfer-optuna", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "source": [
                    "# Private video-transfer Optuna screen\n",
                    "VideoMAE + 3D Swin. No submission. Report-proxy tuning only; no gold evaluation.\n"]},
                    {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": source.splitlines(keepends=True)}]}
    (ROOT / "transfer-optuna.ipynb").write_text(json.dumps(notebook, indent=1))
    metadata = {"id": "sarthaksharma14/rsna-knee-video-transfer-optuna-screen",
                "title": "RSNA Knee Video Transfer Optuna Private Screen",
                "code_file": "transfer-optuna.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "competition_sources": ["rsna-knee-abnormality-detection"],
                "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels"],
                "kernel_sources": ["sarthaksharma14/rsna-knee-288px-12-slice-cache"], "model_sources": []}
    (ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-sha", required=True)
    args = parser.parse_args()
    if len(args.label_sha) != 64 or any(c not in "0123456789abcdef" for c in args.label_sha):
        raise ValueError("expected exact label SHA256")
    build(args.label_sha)
