"""Validate accepted version1, not the newer unlaunched bounded-lookup build."""
import ast
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / 'frozen-0958-control'))
from control import blend, read_preview


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    root = ROOT / f'.audit/r6-adapter-exploratory-v{args.version}'
    out = root / 'output'
    source = root / 'source/rsna-knee-r6-adapter-exploratory-submission.ipynb'
    nb = json.loads(source.read_text(encoding='utf-8'))
    expected = json.loads((HERE / 'adapter-exploratory.ipynb').read_text(encoding='utf-8'))
    assert len(nb['cells']) == len(expected['cells'])
    for i, (actual, current) in enumerate(zip(nb['cells'], expected['cells'])):
        text = ''.join(current['source'])
        if i == 8 and args.version == 1:
            start = text.index("_input_root = Path('/kaggle/input')")
            end = text.index('assert len(_epoch_candidates) == 1')
            text = text[:start] + "_epoch_candidates = list(Path('/kaggle/input').rglob('epoch_3.pt'))\n_epoch_candidates = [_p for _p in _epoch_candidates if _p.parent.name == 'results' and _p.parent.parent.name == 'r6_lowrank_adaptation']\n" + text[end:]
        assert actual['cell_type'] == current['cell_type']
        assert ''.join(actual['source']) == text, f'saved source cell {i} differs'
    adapter = json.loads((out / 'exploratory_adapter_receipt.json').read_text())
    assert adapter['final_epoch_sha256'] == '1b488e45ee6f9905241260b79a2f22a588a29ad5cce765d3d2406a65b7f9aa69'
    assert adapter['adapter_source_sha256'] == '8d9c4265dd13715e417db5c80d0b9d028ce399dcc35d31bd15515b6f417c3fea'
    assert adapter['modified_fold_only'] == 0 and adapter['zero_fallbacks']
    assert adapter['weights_preprocessing_tta_stackers_unchanged']
    assert adapter['training_proxy_status'] == 'REJECTED' and not adapter['improvement_established']
    # Rebuild the exact worker patch from the saved source, without executing it.
    setup = ast.parse(''.join(nb['cells'][8]['source']))
    hook = next(ast.literal_eval(n.value.func.value) for n in setup.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == '_hook' for t in n.targets))
    original = (ROOT / '.audit/rsna-knee-blend-code/r6src/compact_infer.py').read_text(encoding='utf-8')
    anchor = '                m.load_state_dict(sd, strict=True)\n'
    candidates = ['/kaggle/input/rsna-knee-r6-lowrank-spatial-adaptation',
                  '/kaggle/input/notebooks/sarthaksharma14/rsna-knee-r6-lowrank-spatial-adaptation']
    hashes = []
    for mount in candidates:
        path = mount + '/r6_lowrank_adaptation/results/epoch_3.pt'
        modified = original.replace(anchor, anchor + hook.replace('__ADAPTER_PATH__', repr(path)))
        hashes.append(hashlib.sha256(modified.encode()).hexdigest())
    assert adapter['patched_worker_sha256'] in hashes
    ids, r6 = read_preview(out / 'ours_sub.csv')
    old_ids, _ = read_preview(ROOT / '.audit/approved-helios958-v1/ours_sub.csv')
    assert ids == old_ids and len(ids) == adapter['rows'] == 3
    r384 = read_preview(out / 'raptor_raw.csv', ids)[1]
    r224 = read_preview(out / 'raptor224_raw.csv', ids)[1]
    final = read_preview(out / 'submission.csv', ids)[1]
    assert np.array_equal(blend(r6, r384, r224), final.astype(np.float32))
    assert sha(out / 'submission.csv') == adapter['submission_sha256']
    receipt = json.loads((out / 'submission_receipt.json').read_text())
    baseline = json.loads((ROOT / '.audit/approved-helios958-v1/submission_receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['rows'] == receipt['scored'] == 3
    assert receipt['fallback_studies'] == 0 and not receipt['fallback_reasons']
    assert all(v == 0 for v in receipt['worker_exit_codes'].values())
    assert len(receipt['workers']) == 2
    for i, worker in enumerate(receipt['workers']):
        old = baseline['workers'][i]
        assert worker['status'] == 'PASS' and not worker['fallback'] and not worker['timed_out']
        for key in ('models', 'backbone', 'crop', 'crop2', 'views', 'phys_mm', 'anchor_weights_sha256', 'flip_tta'):
            assert worker[key] == old[key], key
        assert worker['scored'] == worker['studies']
        for name, digest in worker['sha256'].items():
            assert digest == (adapter['patched_worker_sha256'] if name == 'compact_infer.py' else old['sha256'][name])
        for name in ('state_stacker', 'fluid_stacker'):
            assert worker[name]['sha256'] == old[name]['sha256'] and not worker[name]['stack_fallback']
        log = (out / f'worker{i}_log.txt').read_text()
        assert 'EXPLORATORY_ADAPTER_FOLD0_LOADED '+adapter['final_epoch_sha256'] in log
        assert 'Traceback (most recent call last)' not in log
    blend_receipt = json.loads((out / 'blend_receipt.json').read_text())
    old_blend = json.loads((ROOT / '.audit/approved-helios958-v1/blend_receipt.json').read_text())
    assert blend_receipt['weights'] == old_blend['weights']
    for key in ('ours_failed', 'raptor384_failed', 'raptor224_failed', 'both_failed'):
        assert blend_receipt[key] == 0
    log = (out / 'rsna-knee-r6-adapter-exploratory-submission.log').read_text()
    for marker in ('RAPTOR_STAGE_DONE', 'RAPTOR224_STAGE_DONE', 'NOTEBOOK_COMPLETE', 'EXPLORATORY_ADAPTER_COMPLETE'):
        assert marker in log
    assert 'Traceback (most recent call last)' not in log and 'BLEND_FALLBACK' not in log
    upload_sha = ('6ce1e157fd7cc50a1ca79b10917a7bd190a73ac659318a8d79dddaecf7b93700' if args.version == 1
                  else '38a08459197fca3f0b4f3abafb41112701cd8eef778a665d950104d1daf70a6d')
    if args.version == 2:
        assert sha(HERE / 'adapter-exploratory.ipynb') == upload_sha
    report = {'version': args.version, 'integrity_passed': True, 'uploaded_notebook_sha256': upload_sha,
        'downloaded_notebook_sha256': sha(source), 'saved_cell_sources_match_accepted_version': True,
        'serialization_note': 'Kaggle reserializes notebook JSON; every cell source/type was independently matched',
        'rows': 3, 'zero_fallbacks': True, 'final_epoch_loaded_both_workers': True,
        'exact_blend_formula_reproduced': True, 'csv_sha256': sha(out / 'submission.csv'),
        'preview_csv_equals_old_baseline': sha(out / 'submission.csv') == sha(ROOT / '.audit/approved-helios958-v1/submission.csv'),
        'scope': 'three-study preview integrity only; equal preview ranks do not establish equal hidden predictions or accuracy',
        'log_warning_free': False, 'improvement_established': False,
        'adapter_receipt': adapter, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    name = 'preview_validation.json' if args.version == 1 else 'v2_preview_validation.json'
    (HERE / name).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
