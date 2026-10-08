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
    {
        "cell_type": "markdown",
        "id": "attribution",
        "metadata": {},
        "source": """# Strict private reproduction of public MAViT 288 reader

Source: `hengck23/mavit-288-baseline-01-1-5sec-study-1gpu` (public Kaggle notebook).  
Assets: `hengck23/hengck23-rnsa-knee` (MIT).  
MAViT lineage: Qihang Fan et al., *Rectifying Magnitude Neglect in Linear Attention*, ICCV 2025, via MALA.  
DICOM fast-reader attribution in the upstream notebook: `nartaa/rsna-knee-0949-anatomical-mirror`.

This copy preserves the upstream model and preprocessing contract while adding pinned asset identities,
strict no-fallback execution, official schema/test-ID validation, and a completion receipt.
""",
    },
    code("""from pathlib import Path
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch

OUT = Path('/kaggle/working')
PUBLIC_NOTEBOOK_SHA256 = '41de28b37abcc1caba51f92a4e7117579ad262b240dc52f881cb5fe25f92d59f'
EXPECTED = {
    '00000008.pth': '7acd2a990a89c7a7d73ec0488badc5058a61b0c77142742c532ebc196749bdbc',
    'configure.py': '3a46d4e6f22171e28909574d98d006154a519363a36f40aba31ea799226f4cee',
    'mavit.py': 'e4a48c8ca5fcc0417d5c051b0c0410a0dc0a34c74f51f3e8bc58285ee363936e',
    'model.py': '082c8c8baf86c3c25f878d2bb5c5e0ba1645600e055810f293bfc3888233e8d3',
    'my_fastread_knee.py': '37140c1f96f3afa186c75820e445619511f349d41a18a3ef75b2cffa7d57efa7',
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def find_unique(name):
    hits = list(Path('/kaggle/input').rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one {name}, found {len(hits)}: {hits[:8]}')
    return hits[0]

assets = {}
for name, expected in EXPECTED.items():
    path = find_unique(name)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f'asset hash mismatch for {name}: {actual}')
    assets[name] = path
asset_root = assets['model.py'].parent
if any(path.parent != asset_root for path in assets.values()):
    raise RuntimeError('public model assets do not share one dataset root')
sys.path.insert(0, str(asset_root))

comp_candidates = [
    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
    Path('/kaggle/input/rsna-knee-abnormality-detection'),
]
comp = next((p for p in comp_candidates if (p / 'test.csv').is_file()), None)
if comp is None:
    raise FileNotFoundError('competition test root not found')
for required in ['test.csv', 'test_series.csv', 'sample_submission.csv']:
    if not (comp / required).is_file():
        raise FileNotFoundError(comp / required)
if not (comp / 'test_series').is_dir():
    raise FileNotFoundError(comp / 'test_series')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required')
print('asset root:', asset_root)
print('competition:', comp)
print('GPU:', torch.cuda.get_device_name(0))
""", "identity"),
    code("""from model import KneeModel
from my_fastread_knee import make_knee_dataset_reader

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
NUM_VIEW = 6
NUM_SLICE = 8
SIZE = 288
DEVICE = 'cuda'

sample = pd.read_csv(comp / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
test = pd.read_csv(comp / 'test.csv', dtype={'StudyInstanceUID': str})
series = pd.read_csv(comp / 'test_series.csv', dtype={
    'StudyInstanceUID': str, 'SeriesInstanceUID': str})
expected_columns = ['StudyInstanceUID', *LABELS]
if sample.columns.tolist() != expected_columns:
    raise RuntimeError(f'official submission schema drift: {sample.columns.tolist()}')
test_ids = test.StudyInstanceUID.tolist()
if sample.StudyInstanceUID.tolist() != test_ids:
    raise RuntimeError('sample submission and test IDs differ')
if not test_ids or len(test_ids) != len(set(test_ids)):
    raise RuntimeError('empty or duplicate hidden test IDs')

model = KneeModel(pretrained=False)
saved = torch.load(assets['00000008.pth'], map_location='cpu', weights_only=True)
if sorted(saved.keys()) != ['epoch', 'model_state_dict'] or int(saved['epoch']) != 8:
    raise RuntimeError(f'checkpoint metadata drift: {saved.keys()} epoch={saved.get("epoch")}')
load_result = model.load_state_dict(saved['model_state_dict'], strict=False)
if load_result.missing_keys or load_result.unexpected_keys:
    raise RuntimeError(f'checkpoint state mismatch: {load_result}')
model = model.to(DEVICE).eval()

reader = make_knee_dataset_reader(
    out_size=SIZE, num_slice=NUM_SLICE,
    percentile=(1.0, 99.0), crop_mm=130.0,
    io_threads=32, crop_only_if_nonsquare=True,
)

predictions = []
failures = []
t0 = time.time()
for i, study_id in enumerate(test_ids):
    study_series = series[series['StudyInstanceUID'] == study_id]
    if study_series.empty:
        raise RuntimeError(f'no series rows for hidden test study {study_id}')
    try:
        item = reader(study_id, study_series, str(comp / 'test_series'), 'valid')
        volume = torch.from_numpy(item['volume']).unsqueeze(0).byte().to(DEVICE)
        view_mask = torch.from_numpy(item['view_mask']).unsqueeze(0).bool().to(DEVICE)
        slice_pos = torch.from_numpy(item['slice']).unsqueeze(0).float().to(DEVICE)
        if volume.shape[1:3] != (NUM_VIEW, NUM_SLICE):
            raise RuntimeError(f'volume shape drift {tuple(volume.shape)}')
        if not view_mask.any():
            raise RuntimeError('all clinical views are empty')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            logits = model.infer(volume, view_mask, slice_pos)
            probability = torch.sigmoid(logits).float().cpu().numpy()
        if probability.shape != (1, len(LABELS)):
            raise RuntimeError(f'probability shape drift {probability.shape}')
        if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
            raise RuntimeError('invalid probabilities')
        predictions.append(probability[0])
    except Exception as exc:
        failures.append({'study_id': study_id, 'error': f'{type(exc).__name__}: {exc}'})
        raise
    if (i + 1) % 100 == 0 or i + 1 == len(test_ids):
        print(f'predicted {i+1}/{len(test_ids)} in {time.time()-t0:.1f}s', flush=True)

prediction = np.asarray(predictions, dtype=np.float32)
if prediction.shape != (len(test_ids), len(LABELS)):
    raise RuntimeError(f'incomplete prediction matrix {prediction.shape}')
submission = pd.DataFrame(prediction, columns=LABELS)
submission.insert(0, 'StudyInstanceUID', test_ids)
if submission.columns.tolist() != expected_columns:
    raise RuntimeError('final schema mismatch')
if submission.StudyInstanceUID.tolist() != test_ids or submission.StudyInstanceUID.duplicated().any():
    raise RuntimeError('final hidden test identity mismatch')
values = submission[LABELS].to_numpy(np.float64)
if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
    raise RuntimeError('final probabilities invalid')
""", "inference"),
    code("""stage = OUT / '_mavit_submission_stage.csv'
submission.to_csv(stage, index=False)
roundtrip = pd.read_csv(stage, dtype={'StudyInstanceUID': str}, float_precision='round_trip')
if roundtrip.columns.tolist() != expected_columns:
    raise RuntimeError('CSV round-trip schema mismatch')
if roundtrip.StudyInstanceUID.tolist() != test_ids:
    raise RuntimeError('CSV round-trip identity mismatch')
if not np.array_equal(roundtrip[LABELS].to_numpy(np.float64), values):
    raise RuntimeError('CSV round-trip numeric mismatch')

receipt = {
    'status': 'COMPLETE',
    'contract': 'strict source-faithful public MAViT 288 reproduction; raw probabilities; no fallback',
    'upstream_notebook': 'hengck23/mavit-288-baseline-01-1-5sec-study-1gpu',
    'upstream_notebook_sha256': PUBLIC_NOTEBOOK_SHA256,
    'dataset': 'hengck23/hengck23-rnsa-knee',
    'dataset_license': 'MIT',
    'asset_hashes': EXPECTED,
    'checkpoint_epoch': 8,
    'study_count': len(test_ids),
    'unique_study_count': len(set(test_ids)),
    'series_count': len(series),
    'prediction_shape': list(prediction.shape),
    'probability_min': float(values.min()),
    'probability_max': float(values.max()),
    'fallback_count': 0,
    'failures': failures,
    'elapsed_seconds': time.time() - t0,
    'submission_sha256': sha256(stage),
}
(OUT / 'mavit_reproduction_receipt.json').write_text(json.dumps(receipt, indent=2))
final = OUT / 'submission.csv'
os.replace(stage, final)
if sha256(final) != receipt['submission_sha256']:
    raise RuntimeError('published submission hash changed')
print(json.dumps(receipt, indent=2))
print('published', final)
""", "publish"),
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
(ROOT / "mavit-public-reproduction.ipynb").write_text(json.dumps(notebook, indent=1))

metadata = {
    "id": "sarthaksharma14/rsna-knee-mavit-public-reproduction",
    "title": "RSNA Knee MAViT Public Reproduction",
    "code_file": "mavit-public-reproduction.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["hengck23/hengck23-rnsa-knee"],
    "kernel_sources": [],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
