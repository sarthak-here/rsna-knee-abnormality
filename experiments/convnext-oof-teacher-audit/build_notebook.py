import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cell-{len(source):x}-{abs(hash(source)) & 0xffff:x}",
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    cell("""from pathlib import Path
import hashlib, importlib, json, os, re, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

OUT = Path('/kaggle/working')
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', \"Baker's\", 'Contusion', 'Fracture']
EXPECTED = {
    'cnxt_v0_fold0.pt': '0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80',
    'cnxt_v0_fold1.pt': '61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb',
    'cnxt_v0_fold2.pt': 'f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6',
    'infer.py': '39d4517cc33c4bb1e4a03cfb49a0673a8789055de8c3cc0957c6e3035f6be4af',
    'knee.py': '314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1',
    'make_labels.py': 'cc51b8f609a0aa88eb41b864df502f4f6138e58b6fe567a1880fc714fb8d423c',
    'oof.py': 'f324bfdfb8d41bcc953bf79cd30297bca9a3660666be46c288ad0a67fded5337',
    'preprocess.py': '68601ad84ec3a0f030b5c2d37cc0d657cd3e12b17f7130e7385fba9648d324b6',
    'train.py': 'e41b26488ea5516c113f473c35f2391e0e97dc693b53d076f81b8953f169db2b',
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def find_unique(name):
    hits = list(Path('/kaggle/input').rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f'expected one {name}, found {len(hits)}: {hits[:8]}')
    return hits[0]

train_csv = find_unique('train.csv')
comp = train_csv.parent
if not (comp / 'train_series.csv').is_file() or not (comp / 'train_series').is_dir():
    raise RuntimeError(f'competition layout incomplete under {comp}')

assets = {}
for name, expected in EXPECTED.items():
    p = find_unique(name)
    actual = sha256(p)
    if actual != expected:
        raise RuntimeError(f'hash mismatch for {name}: {actual}')
    assets[name] = p
src_dir = assets['knee.py'].parent
if any(p.parent != src_dir for n, p in assets.items() if n.endswith('.py')):
    raise RuntimeError('public source files are not from one dataset directory')
sys.path.insert(0, str(src_dir))
print('competition:', comp)
print('public source:', src_dir)
print('CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise RuntimeError('GPU is required')
"""),
    cell("""# Reproduce the author's exact report-group folds. Gold rows are never assigned a fold.
tr = pd.read_csv(train_csv)
gold = tr[LABELS].notna().all(axis=1).to_numpy()
weak_idx = np.flatnonzero(~gold)
groups = tr.loc[~gold, 'Report'].map(lambda x: re.sub(r'\\W+', '', str(x).lower())).to_numpy()
order = np.random.RandomState(0).permutation(len(weak_idx))
weak_folds = np.zeros(len(weak_idx), dtype=np.int64)
for fold, (_, va) in enumerate(GroupKFold(5).split(order, groups=groups[order])):
    weak_folds[order[va]] = fold
folds = np.full(len(tr), -1, dtype=np.int64)
folds[weak_idx] = weak_folds
tr['_fold'] = folds

checkpoint_meta = {}
for fold in range(3):
    name = f'cnxt_v0_fold{fold}.pt'
    ck = torch.load(assets[name], map_location='cpu', weights_only=True)
    args = ck.get('args', {})
    if args.get('fold') != fold or args.get('labels') != 'labels_v0':
        raise RuntimeError(f'checkpoint metadata mismatch: {name} {args}')
    if args.get('backbone') != 'convnext_tiny.fb_in22k_ft_in1k' or args.get('res') != 256 or args.get('k_eval') != 12:
        raise RuntimeError(f'checkpoint contract mismatch: {name} {args}')
    checkpoint_meta[name] = args
    del ck

fold_counts = tr.loc[~gold, '_fold'].value_counts().sort_index().astype(int).to_dict()
covered = tr['_fold'].isin([0, 1, 2])
if covered[gold].any() or covered.sum() != sum(fold_counts[k] for k in (0, 1, 2)):
    raise RuntimeError('OOF coverage/gold isolation preflight failed')
print('gold:', int(gold.sum()), 'weak:', int((~gold).sum()), 'folds:', fold_counts)
print('OOF coverage folds 0-2:', int(covered.sum()), f'({covered.sum()/(~gold).sum():.1%})')
"""),
    cell("""# Source-faithful DICOM preprocessing and model inference, one held-out fold per checkpoint.
import infer
from knee import KneeNet

def safe_load_models(ckpts, device='cuda'):
    models = []
    for path in ckpts:
        ck = torch.load(path, map_location='cpu', weights_only=True)
        model = KneeNet(ck['args']['backbone'].split('.')[0], pretrained=False)
        model.load_state_dict(ck['model'], strict=True)
        model = model.to(device).to(memory_format=torch.channels_last).eval()
        models.append((model, ck['args']['res']))
    return models

infer.load_models = safe_load_models
parts = []
timings = {}
for fold in range(3):
    studies = tr.loc[tr['_fold'] == fold, 'StudyInstanceUID'].tolist()
    if set(studies) & set(tr.loc[gold, 'StudyInstanceUID']):
        raise RuntimeError(f'gold leakage into fold {fold}')
    t0 = time.time()
    pred = infer.run(str(comp), 'train', [str(assets[f'cnxt_v0_fold{fold}.pt'])],
                     k=12, studies=studies, workers=4, bs=4)
    if pred.index.tolist() != studies or pred.shape != (len(studies), len(LABELS)):
        raise RuntimeError(f'prediction identity/shape failure for fold {fold}')
    if not np.isfinite(pred.to_numpy()).all() or not ((pred.to_numpy() >= 0) & (pred.to_numpy() <= 1)).all():
        raise RuntimeError(f'non-finite/out-of-range prediction in fold {fold}')
    pred.insert(0, 'teacher_fold', fold)
    parts.append(pred.reset_index())
    timings[str(fold)] = time.time() - t0
    print(f'fold {fold}: {len(studies)} held-out studies in {timings[str(fold)]:.1f}s', flush=True)
    del pred
    torch.cuda.empty_cache()

oof = pd.concat(parts, ignore_index=True)
if len(oof) != covered.sum() or oof.StudyInstanceUID.nunique() != len(oof):
    raise RuntimeError('OOF output has missing or duplicate studies')
expected_ids = set(tr.loc[covered, 'StudyInstanceUID'])
if set(oof.StudyInstanceUID) != expected_ids:
    raise RuntimeError('OOF identity mismatch')
if set(oof.StudyInstanceUID) & set(tr.loc[gold, 'StudyInstanceUID']):
    raise RuntimeError('gold study found in OOF output')
oof.to_csv(OUT / 'convnext_oof_predictions.csv', index=False)
"""),
    cell("""report = {
    'status': 'COMPLETE',
    'purpose': 'inference-only held-out teacher audit; no competition predictions',
    'source': 'goodpjw2008/rsna-knee-2-5d-convnext-reader',
    'license': 'Apache-2.0',
    'checkpoint_hashes': {n: EXPECTED[n] for n in EXPECTED if n.endswith('.pt')},
    'source_hashes': {n: EXPECTED[n] for n in EXPECTED if n.endswith('.py')},
    'checkpoint_metadata': checkpoint_meta,
    'fold_counts_all_weak': {str(k): v for k, v in fold_counts.items()},
    'teacher_folds': [0, 1, 2],
    'weak_study_count': int((~gold).sum()),
    'gold_study_count': int(gold.sum()),
    'oof_study_count': int(len(oof)),
    'oof_coverage_fraction': float(len(oof) / (~gold).sum()),
    'oof_unique_study_count': int(oof.StudyInstanceUID.nunique()),
    'gold_studies_in_oof': 0,
    'timings_seconds': timings,
    'prediction_file_sha256': sha256(OUT / 'convnext_oof_predictions.csv'),
    'submission_created': False,
}
(OUT / 'convnext_oof_audit.json').write_text(json.dumps(report, indent=2))
for forbidden in ('submission.csv', 'submission.parquet'):
    if (OUT / forbidden).exists():
        raise RuntimeError(f'forbidden competition artifact created: {forbidden}')
print(json.dumps(report, indent=2))
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "convnext-oof-teacher-audit.ipynb").write_text(json.dumps(notebook, indent=1))

metadata = {
    "id": "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit",
    "title": "RSNA Knee ConvNeXt OOF Teacher Audit",
    "code_file": "convnext-oof-teacher-audit.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader"],
    "kernel_sources": [],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

