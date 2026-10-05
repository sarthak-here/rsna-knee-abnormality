import json
from pathlib import Path


root = Path(__file__).resolve().parent
code = r'''
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SOURCE = Path('/kaggle/input/rsna-knee-288px-12-slice-cache')
H5 = SOURCE / 'train_288x12.h5'
META = SOURCE / 'highres_cache_meta.json'
STUDIES = SOURCE / 'highres_cache_studies.csv'
REPORT = Path('/kaggle/working/highres_cache_validation.json')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


for path in (H5, META, STUDIES):
    if not path.is_file():
        raise FileNotFoundError(path)

meta = json.loads(META.read_text())
studies = pd.read_csv(STUDIES)
if list(studies.columns) != ['StudyInstanceUID']:
    raise RuntimeError(f'unexpected study columns: {studies.columns.tolist()}')
if len(studies) != 4407 or not studies['StudyInstanceUID'].is_unique:
    raise RuntimeError('study identity/count validation failed')

study_sha = digest(STUDIES)
if study_sha != meta['studies_sha256']:
    raise RuntimeError('study manifest SHA-256 mismatch')
if H5.stat().st_size != meta['h5_bytes']:
    raise RuntimeError('HDF5 byte-size mismatch')
h5_sha = digest(H5)
if h5_sha != meta['h5_sha256']:
    raise RuntimeError('HDF5 SHA-256 mismatch')

expected_shape = (4407, 6, 12, 288, 288)
pixel_min, pixel_max = 255, 0
occupied_slots = 0
empty_slots = 0
occupied_all_zero = 0
with h5py.File(H5, 'r') as handle:
    if set(handle.keys()) != {'pixels', 'masks'}:
        raise RuntimeError(f'unexpected HDF5 keys: {list(handle.keys())}')
    pixels = handle['pixels']
    masks = handle['masks']
    if pixels.shape != expected_shape or pixels.dtype != np.dtype('uint8'):
        raise RuntimeError(f'pixel contract mismatch: {pixels.shape} {pixels.dtype}')
    if pixels.chunks != (1, 1, 12, 288, 288):
        raise RuntimeError(f'pixel chunk mismatch: {pixels.chunks}')
    if pixels.compression != 'gzip' or pixels.compression_opts != 2 or not pixels.shuffle:
        raise RuntimeError('pixel compression contract mismatch')
    if masks.shape != (4407, 6) or masks.dtype != np.dtype('float32'):
        raise RuntimeError(f'mask contract mismatch: {masks.shape} {masks.dtype}')
    for start in range(0, 4407, 64):
        stop = min(start + 64, 4407)
        mask_block = masks[start:stop]
        if not np.isfinite(mask_block).all() or not np.isin(mask_block, (0.0, 1.0)).all():
            raise RuntimeError(f'invalid masks in studies {start}:{stop}')
        pixel_block = pixels[start:stop]
        pixel_min = min(pixel_min, int(pixel_block.min()))
        pixel_max = max(pixel_max, int(pixel_block.max()))
        slot_nonzero = np.any(pixel_block != 0, axis=(2, 3, 4))
        occupied = mask_block == 1.0
        occupied_all_zero += int(np.logical_and(occupied, ~slot_nonzero).sum())
        occupied_slots += int(occupied.sum())
        empty_slots += int((~occupied).sum())

if occupied_all_zero:
    raise RuntimeError(f'{occupied_all_zero} occupied slots contain only zero pixels')

report = {
    'status': 'passed',
    'contract': meta['contract'],
    'shape': list(expected_shape),
    'dtype': 'uint8',
    'chunks': [1, 1, 12, 288, 288],
    'compression': 'gzip-2-shuffle',
    'study_count': len(studies),
    'unique_studies': int(studies['StudyInstanceUID'].nunique()),
    'occupied_slots': occupied_slots,
    'empty_slots': empty_slots,
    'occupied_all_zero': occupied_all_zero,
    'pixel_min': pixel_min,
    'pixel_max': pixel_max,
    'h5_bytes': H5.stat().st_size,
    'h5_sha256': h5_sha,
    'studies_sha256': study_sha,
    'source_kernel': 'sarthaksharma14/rsna-knee-288px-12-slice-cache',
    'creates_submission': False,
}
REPORT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
'''

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "purpose",
            "metadata": {},
            "source": [
                "# Strict validation: RSNA Knee 288px / 12-slice cache\n",
                "Validates the complete private HDF5 artifact without creating predictions or submissions.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "validate",
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(keepends=True),
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / 'highres-cache-validation.ipynb').write_text(json.dumps(notebook), encoding='utf-8')
