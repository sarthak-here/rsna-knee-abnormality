import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = json.loads((root / "source-public-cache.ipynb").read_text(encoding="utf-8"))
for index, cell in enumerate(source["cells"]):
    cell.setdefault("id", f"cell-{index:02d}")

# Keep the attributed public implementation cells, but make the resolution/sampling
# decision explicit before CFG is defined.
source["cells"][3]["source"] = [
    "import os\n",
    "os.environ['RSNA_IMG'] = '288'\n",
    "os.environ['RSNA_CROP_MM'] = '140'\n",
    "os.environ['RSNA_N_GROUP_MAX'] = '4'\n",
    "os.environ['RSNA_WINDOW'] = '0.08,0.92'\n",
    "for k, v in os.environ.items():\n",
    "    if k.startswith('RSNA_'):\n",
    "        print(f'{k} = {v}')\n",
]

writer = r'''
import json
import h5py

OUT = Path('/kaggle/working/train_288x12.h5')
META = Path('/kaggle/working/highres_cache_meta.json')
STUDIES = Path('/kaggle/working/highres_cache_studies.csv')

root = find_root()
train_series = pd.read_csv(root / 'train_series.csv')
plane_map = dict(zip(train_series['SeriesInstanceUID'], train_series['Anatomical_Plane']))
log('highres header pass: train')
headers = annotate(walk(root, 'train_series'))
if not headers.empty:
    headers['plane'] = headers['SeriesInstanceUID'].map(plane_map)
laterality = {study: resolve_laterality(group)
              for study, group in headers.groupby('StudyInstanceUID')}
slot_map = pick_slots(headers, plane_map)
all_studies = sorted(slot_map)
if len(all_studies) != 4407:
    raise RuntimeError(f'unexpected highres study count {len(all_studies)}')

n_group = 4
n_slice = CFG.group * n_group
if CFG.img != 288 or CFG.crop_mm != 140 or n_slice != 12:
    raise RuntimeError('highres cache config drift')

pd.DataFrame({'StudyInstanceUID': all_studies}).to_csv(STUDIES, index=False)
with h5py.File(OUT, 'w') as handle:
    pixels = handle.create_dataset(
        'pixels', shape=(len(all_studies), len(SLOTS), n_slice, CFG.img, CFG.img),
        dtype='u1', chunks=(1, 1, n_slice, CFG.img, CFG.img),
        compression='gzip', compression_opts=2, shuffle=True)
    masks = handle.create_dataset('masks', shape=(len(all_studies), len(SLOTS)),
                                  dtype='f4', chunks=(256, len(SLOTS)))
    bounds = np.linspace(0, len(all_studies), 5).astype(int)
    for shard in range(4):
        lo, hi = int(bounds[shard]), int(bounds[shard + 1])
        subset_ids = all_studies[lo:hi]
        subset = {uid: slot_map[uid] for uid in subset_ids}
        studies, cache, mask = build_cache(
            subset, plane_map, laterality, f'highres-s{shard:02d}', n_group)
        if studies != subset_ids:
            raise RuntimeError(f'shard {shard} identity drift')
        expected = sum(len(slot_map[uid]) for uid in subset_ids)
        if int(mask.sum()) < expected:
            raise RuntimeError(f'shard {shard} incomplete {int(mask.sum())}/{expected}')
        pixels[lo:hi] = cache
        masks[lo:hi] = mask
        handle.flush()
        log(f'highres shard {shard}: wrote studies {lo}:{hi}')
        del cache, mask
        gc.collect()

import hashlib
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

meta = {
    'contract': 'training pixels only; no labels; no submission',
    'shape': [len(all_studies), len(SLOTS), n_slice, CFG.img, CFG.img],
    'crop_mm': CFG.crop_mm,
    'window': os.environ['RSNA_WINDOW'],
    'compression': 'HDF5 gzip level 2, one-study/one-slot chunks, shuffle',
    'study_count': len(all_studies),
    'h5_bytes': OUT.stat().st_size,
    'h5_sha256': digest(OUT),
    'studies_sha256': digest(STUDIES),
    'source': 'stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache',
}
META.write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta, indent=2))
'''

source["cells"][25]["source"] = writer.splitlines(keepends=True)
source["cells"][25]["outputs"] = []
source["cells"][25]["execution_count"] = None
source["cells"][0]["source"] = [
    "# RSNA Knee 288px / 12-slice physical cache\n",
    "Derived from Steven Lee Hans' attributed public CPU cache notebook. This run changes\n",
    "the pixel contract to 288px, 140mm, 12 slices and writes random-access HDF5.\n",
]
(root / "highres-pixel-cache.ipynb").write_text(json.dumps(source), encoding="utf-8")
