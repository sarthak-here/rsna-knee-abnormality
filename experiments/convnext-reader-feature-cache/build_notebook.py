import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def code(source):
    return {"cell_type": "code", "execution_count": None,
            "id": f"cache-{len(source):x}", "metadata": {},
            "outputs": [], "source": source}


cells = [
    code("""from pathlib import Path
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch

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
print('GPU:', torch.cuda.get_device_name(0), 'bf16 native:', torch.cuda.is_bf16_supported())
"""),
    code("""from infer import DicomStudyDataset
from knee import LABELS, N_SLOTS, KneeNet, build_study_table, make_views
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
encoder = model.enc.cuda().to(memory_format=torch.channels_last).eval()
feature_dim = int(encoder.num_features)
del model, ck

train = pd.read_csv(train_csv, dtype={'StudyInstanceUID': str})
studies = train.StudyInstanceUID.tolist()
if len(studies) != 4407 or len(set(studies)) != len(studies):
    raise RuntimeError('study identity drift')
series = pd.read_csv(comp / 'train_series.csv', dtype={
    'StudyInstanceUID': str, 'SeriesInstanceUID': str})
n_files = {}
for st, sr in zip(series.StudyInstanceUID, series.SeriesInstanceUID):
    directory = comp / 'train_series' / st / sr
    n_files[sr] = min(len(os.listdir(directory)), MAX_SLICES) if directory.is_dir() else 0
table = build_study_table(series, n_files)
dataset = DicomStudyDataset(studies, table, str(comp / 'train_series'), k=12)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True)

features = np.zeros((len(studies), N_SLOTS, 12, feature_dim), dtype=np.float16)
masks = np.zeros((len(studies), N_SLOTS), dtype=np.uint8)
positions = np.zeros((len(studies), N_SLOTS, 12), dtype=np.float16)
start = 0
t0 = time.time()
with torch.inference_mode():
    for step, (x, mask, pos) in enumerate(loader):
        batch, slots, windows = x.shape[:3]
        keep = mask.view(-1)
        if keep.any():
            xv = x.view(batch * slots, windows, *x.shape[3:])[keep]
            xv = xv.flatten(0, 1).cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                encoded = encoder(make_views(
                    xv, args['res'], False, windows).contiguous(
                        memory_format=torch.channels_last))
            packed = torch.zeros(batch * slots, windows, feature_dim,
                                 dtype=torch.float16)
            packed[keep] = encoded.reshape(-1, windows, feature_dim).cpu()
            packed = packed.view(batch, slots, windows, feature_dim).numpy()
        else:
            packed = np.zeros((batch, slots, windows, feature_dim), np.float16)
        features[start:start + batch] = packed
        masks[start:start + batch] = mask.numpy().astype(np.uint8)
        positions[start:start + batch] = pos.numpy().astype(np.float16)
        start += batch
        if step % 50 == 0:
            print(f'encoded {start}/{len(studies)} in {time.time()-t0:.1f}s', flush=True)

if start != len(studies):
    raise RuntimeError(f'incomplete feature extraction: {start}/{len(studies)}')
if not np.isfinite(features).all() or not np.isfinite(positions).all():
    raise RuntimeError('non-finite feature package')
if not np.isin(masks, [0, 1]).all() or (masks.sum(axis=1) == 0).any():
    raise RuntimeError('invalid or empty study masks')
occupied = masks.astype(bool)[:, :, None, None]
if np.any(np.abs(features).sum(axis=(2, 3))[masks.astype(bool)] == 0):
    raise RuntimeError('all-zero occupied slot features')
"""),
    code("""feature_path = OUT / 'convnext_reader_fold0_features.npz'
np.savez_compressed(feature_path,
    ids=np.asarray(studies), features=features, masks=masks, positions=positions,
    checkpoint_sha256=np.asarray(EXPECTED['cnxt_v0_fold0.pt']),
    checkpoint_fold=np.asarray(0), source_dataset=np.asarray(
        'goodpjw2008/rsna-knee-2-5d-convnext-reader'))
report = {
    'status': 'COMPLETE',
    'contract': 'public ConvNeXt fold0 representation cache; no fitting and no submission',
    'source': 'goodpjw2008/rsna-knee-2-5d-convnext-reader',
    'license': 'Apache-2.0',
    'checkpoint_sha256': EXPECTED['cnxt_v0_fold0.pt'],
    'source_hashes': {k: v for k, v in EXPECTED.items() if k.endswith('.py')},
    'checkpoint_args': args,
    'study_count': len(studies),
    'unique_study_count': len(set(studies)),
    'feature_shape': list(features.shape),
    'feature_dtype': str(features.dtype),
    'mask_shape': list(masks.shape),
    'position_shape': list(positions.shape),
    'occupied_slot_count': int(masks.sum()),
    'empty_study_count': int((masks.sum(axis=1) == 0).sum()),
    'finite': True,
    'elapsed_seconds': time.time() - t0,
    'feature_file_bytes': feature_path.stat().st_size,
    'feature_file_sha256': sha256(feature_path),
    'submission_created': False,
}
(OUT / 'convnext_reader_feature_report.json').write_text(json.dumps(report, indent=2))
if (OUT / 'submission.csv').exists():
    raise RuntimeError('forbidden submission artifact')
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
(ROOT / "convnext-reader-feature-cache.ipynb").write_text(json.dumps(notebook, indent=1))

metadata = {
    "id": "sarthaksharma14/rsna-knee-convnext-reader-feature-cache",
    "title": "RSNA Knee ConvNeXt Reader Feature Cache",
    "code_file": "convnext-reader-feature-cache.ipynb",
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

