import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def code(source, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    code("""from pathlib import Path
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

OUT = Path('/kaggle/working')
EXPECTED = {
    'cnxt_v0_fold0.pt': '0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80',
    'infer.py': '39d4517cc33c4bb1e4a03cfb49a0673a8789055de8c3cc0957c6e3035f6be4af',
    'knee.py': '314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1',
    'preprocess.py': '68601ad84ec3a0f030b5c2d37cc0d657cd3e12b17f7130e7385fba9648d324b6',
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
    path = find_unique(name)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f'hash mismatch for {name}: {actual}')
    assets[name] = path
src_dir = assets['knee.py'].parent
if assets['infer.py'].parent != src_dir or assets['preprocess.py'].parent != src_dir:
    raise RuntimeError('source files do not share a dataset directory')
sys.path.insert(0, str(src_dir))
if not torch.cuda.is_available():
    raise RuntimeError('GPU required')
print('competition:', comp)
print('source:', src_dir)
print('GPU:', torch.cuda.get_device_name(0))
""", "setup"),
    code("""from infer import DicomStudyDataset
from knee import LABELS, N_SLOTS, KneeNet, build_study_table
from preprocess import MAX_SLICES

ck = torch.load(assets['cnxt_v0_fold0.pt'], map_location='cpu', weights_only=True)
args = ck['args']
expected_args = {'fold': 0, 'labels': 'labels_v0',
                 'backbone': 'convnext_tiny.fb_in22k_ft_in1k',
                 'res': 256, 'k_eval': 12}
for key, value in expected_args.items():
    if args.get(key) != value:
        raise RuntimeError(f'checkpoint contract mismatch: {key}={args.get(key)}')
model = KneeNet(args['backbone'].split('.')[0], pretrained=False)
model.load_state_dict(ck['model'], strict=True)
model = model.cuda().eval()
del ck

train = pd.read_csv(train_csv, dtype={'StudyInstanceUID': str})
missing = [c for c in LABELS if c not in train.columns]
if missing:
    raise RuntimeError(f'missing organizer labels: {missing}')
gold_mask = train[LABELS].notna().all(axis=1)
gold = train.loc[gold_mask, ['StudyInstanceUID', *LABELS]].copy()
gold_ids = gold.StudyInstanceUID.tolist()
gold_y = gold[LABELS].to_numpy(np.int64)
if len(gold_ids) != 58 or len(set(gold_ids)) != 58:
    raise RuntimeError(f'organizer gold identity drift: {len(gold_ids)}')

series = pd.read_csv(comp / 'train_series.csv', dtype={
    'StudyInstanceUID': str, 'SeriesInstanceUID': str})
n_files = {}
for st, sr in zip(series.StudyInstanceUID, series.SeriesInstanceUID):
    directory = comp / 'train_series' / st / sr
    n_files[sr] = min(len(os.listdir(directory)), MAX_SLICES) if directory.is_dir() else 0
table = build_study_table(series, n_files)
dataset = DicomStudyDataset(gold_ids, table, str(comp / 'train_series'), k=12)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True)

def anatomical_mirror(x):
    # x [B,6,K,3,H,W]. Canonical public preprocessing defines slots
    # 0/3 as sagittal, 1/4 coronal, 2/5 axial. Mirroring a sagittal
    # triplet reverses its slice channels; coronal/axial views flip width.
    y = x.clone()
    y[:, [0, 3]] = y[:, [0, 3]].flip(3)
    y[:, [1, 2, 4, 5]] = y[:, [1, 2, 4, 5]].flip(-1)
    return y

plain_parts, mirror_parts = [], []
t0 = time.time()
with torch.inference_mode():
    for step, (x, mask, pos) in enumerate(loader):
        x = x.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        pos = pos.cuda(non_blocking=True)
        xm = anatomical_mirror(x)
        with torch.autocast('cuda', dtype=torch.float16):
            plain = torch.sigmoid(model(x, mask, pos, res=args['res'], train=False))
            mirrored = torch.sigmoid(model(xm, mask, pos, res=args['res'], train=False))
        plain_parts.append(plain.float().cpu().numpy())
        mirror_parts.append(mirrored.float().cpu().numpy())
        if step % 5 == 0:
            print(f'evaluated {min((step+1)*2, len(gold_ids))}/{len(gold_ids)}', flush=True)

plain = np.concatenate(plain_parts)
mirrored = np.concatenate(mirror_parts)
averaged = 0.5 * (plain + mirrored)
for name, arr in [('plain', plain), ('mirrored', mirrored), ('averaged', averaged)]:
    if arr.shape != (58, len(LABELS)) or not np.isfinite(arr).all():
        raise RuntimeError(f'{name} prediction contract failed: {arr.shape}')
    if (arr < 0).any() or (arr > 1).any():
        raise RuntimeError(f'{name} predictions outside [0,1]')

def per_target_auc(pred):
    return [float(roc_auc_score(gold_y[:, j], pred[:, j])) for j in range(len(LABELS))]

plain_auc = per_target_auc(plain)
mirror_auc = per_target_auc(mirrored)
average_auc = per_target_auc(averaged)
plain_macro = float(np.mean(plain_auc))
mirror_macro = float(np.mean(mirror_auc))
average_macro = float(np.mean(average_auc))
delta = average_macro - plain_macro

rng = np.random.default_rng(20261008)
boot = []
for _ in range(5000):
    idx = rng.integers(0, len(gold_ids), len(gold_ids))
    diffs = []
    for j in range(len(LABELS)):
        if np.unique(gold_y[idx, j]).size < 2:
            continue
        diffs.append(roc_auc_score(gold_y[idx, j], averaged[idx, j]) -
                     roc_auc_score(gold_y[idx, j], plain[idx, j]))
    if diffs:
        boot.append(float(np.mean(diffs)))
boot = np.asarray(boot)
p_improve = float((boot > 0).mean())
q025, q975 = [float(v) for v in np.quantile(boot, [0.025, 0.975])]
""", "evaluate"),
    code("""pred_path = OUT / 'convnext_mirror_gold_predictions.npz'
np.savez_compressed(pred_path, ids=np.asarray(gold_ids), labels=np.asarray(LABELS),
                    truth=gold_y, plain=plain, mirrored=mirrored, averaged=averaged)
report = {
    'status': 'COMPLETE',
    'contract': 'fixed 50/50 anatomical-mirror TTA diagnostic on untouched organizer gold; no fitting and no submission',
    'source': 'goodpjw2008/rsna-knee-2-5d-convnext-reader',
    'mirror_method_source': 'nartaa/rsna-knee-0949-anatomical-mirror',
    'license': 'Apache-2.0 public ConvNeXt reader; mirror method inspected from public notebook',
    'checkpoint_sha256': EXPECTED['cnxt_v0_fold0.pt'],
    'source_hashes': {k: v for k, v in EXPECTED.items() if k.endswith('.py')},
    'checkpoint_args': args,
    'gold_count': len(gold_ids),
    'gold_fit_count': 0,
    'plain_macro_auc': plain_macro,
    'mirrored_macro_auc': mirror_macro,
    'average_macro_auc': average_macro,
    'average_minus_plain': delta,
    'per_target_plain_auc': dict(zip(LABELS, plain_auc)),
    'per_target_mirror_auc': dict(zip(LABELS, mirror_auc)),
    'per_target_average_auc': dict(zip(LABELS, average_auc)),
    'bootstrap_replicates': int(len(boot)),
    'bootstrap_p_improve': p_improve,
    'bootstrap_q025': q025,
    'bootstrap_q975': q975,
    'predeclared_pass_gate': {
        'delta_min': 0.005,
        'p_improve_min': 0.95,
        'q025_positive': True,
        'average_macro_auc_min': 0.914,
    },
    'pass': bool(delta >= 0.005 and p_improve >= 0.95 and q025 > 0 and average_macro >= 0.914),
    'prediction_file_bytes': pred_path.stat().st_size,
    'prediction_file_sha256': sha256(pred_path),
    'elapsed_seconds': time.time() - t0,
    'submission_created': False,
}
(OUT / 'convnext_mirror_validation_report.json').write_text(json.dumps(report, indent=2))
if (OUT / 'submission.csv').exists():
    raise RuntimeError('forbidden submission artifact')
print(json.dumps(report, indent=2))
""", "report"),
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
(ROOT / "convnext-reader-mirror-validation.ipynb").write_text(json.dumps(notebook, indent=1))

metadata = {
    "id": "sarthaksharma14/rsna-knee-convnext-reader-mirror-validation",
    "title": "RSNA Knee ConvNeXt Reader Mirror Validation",
    "code_file": "convnext-reader-mirror-validation.ipynb",
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
