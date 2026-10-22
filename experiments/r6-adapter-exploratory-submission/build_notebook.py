"""Freeze one final-checkpoint exploratory inference contract, without tuning."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE_SHA = 'ad0b97aca3361f16e32b0836655dd68bc2a37ff0cc5a44f6d6f4de07880d986d'
EPOCH_SHA = '1b488e45ee6f9905241260b79a2f22a588a29ad5cce765d3d2406a65b7f9aa69'
ADAPTER_SHA = '8d9c4265dd13715e417db5c80d0b9d028ce399dcc35d31bd15515b6f417c3fea'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def cell(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.splitlines(True)}


def build():
    base = ROOT / '.audit/final24h-helios-exact/blend-mine-in-private-lb-luck-to-all-upvoters.ipynb'
    assert sha(base.read_bytes()) == BASE_SHA
    nb = json.loads(base.read_text(encoding='utf-8'))
    adapters = (HERE.parent / 'r6-lowrank-spatial-adaptation/adapters.py').read_bytes()
    assert sha(adapters) == ADAPTER_SHA
    checkpoint = ROOT / '.audit/r6-lowrank-v1/output/r6_lowrank_adaptation/results/epoch_3.pt'
    assert sha(checkpoint.read_bytes()) == EPOCH_SHA
    # This runs inside the worker after the original model has been strictly loaded.
    hook = '''                if p.name == 'E_lrprior_s0_f0.pt':
                    import hashlib as _hashlib
                    from adapters import install as _install_adapter
                    _path = __ADAPTER_PATH__
                    with open(_path, 'rb') as _f:
                        assert _hashlib.file_digest(_f, 'sha256').hexdigest() == '__EPOCH_SHA__'
                    _payload = torch.load(_path, map_location='cpu', weights_only=True)
                    assert _payload['epoch'] == 3 and _payload['adapter_rank'] == 8
                    assert _payload['initial_checkpoint_sha256'] == '59cc88a197bc80c7fa5faa89e69ae7bec04f69cab5259a23a0d0ceb9431e53f6'
                    _adapted = _payload['state_dict']
                    for _name, _tensor in sd.items():
                        _key = _name.replace('.mlp.fc2.', '.mlp.fc2.base.') if _name.startswith('enc.stages.3.blocks.') else _name
                        assert _key in _adapted and _tensor.dtype == _adapted[_key].dtype and torch.equal(_tensor, _adapted[_key]), _name
                    _names = _install_adapter(m)
                    assert len(_names) == 6
                    m.load_state_dict(_adapted, strict=True)
                    assert all(not _t.is_floating_point() or torch.isfinite(_t).all() for _t in m.state_dict().values())
                    print('EXPLORATORY_ADAPTER_FOLD0_LOADED', '__EPOCH_SHA__', flush=True)
'''.replace('__EPOCH_SHA__', EPOCH_SHA)
    setup = f'''# Explicit user-authorized exploration; failed proxy gate remains failed.
_input_root = Path('/kaggle/input')
_training_slug = 'rsna-knee-r6-lowrank-spatial-adaptation'
_mounts = [_input_root / _training_slug, _input_root / 'notebooks' / 'sarthaksharma14' / _training_slug]
_mounts += list(_input_root.glob('notebooks/*/' + _training_slug))
_mounts += list(_input_root.glob('*/' + _training_slug))
_epoch_candidates = sorted({{_m / 'r6_lowrank_adaptation' / 'results' / 'epoch_3.pt'
    for _m in _mounts if (_m / 'r6_lowrank_adaptation' / 'results' / 'epoch_3.pt').is_file()}})
assert len(_epoch_candidates) == 1, _epoch_candidates
_epoch_path = _epoch_candidates[0]
assert hashlib.sha256(_epoch_path.read_bytes()).hexdigest() == {EPOCH_SHA!r}
_adapter_source = {adapters.decode('utf-8')!r}
(SRC / 'adapters.py').write_bytes(_adapter_source.encode('utf-8'))
assert hashlib.sha256((SRC / 'adapters.py').read_bytes()).hexdigest() == {ADAPTER_SHA!r}
_infer_path = SRC / 'compact_infer.py'
_infer = _infer_path.read_text()
assert hashlib.sha256(_infer_path.read_bytes()).hexdigest() == SOURCE_SHA['compact_infer.py']
_anchor = '                m.load_state_dict(sd, strict=True)\\n'
assert _infer.count(_anchor) == 1
_hook = {hook!r}.replace('__ADAPTER_PATH__', repr(str(_epoch_path)))
_infer = _infer.replace(_anchor, _anchor + _hook)
compile(_infer, str(_infer_path), 'exec')
_infer_path.write_bytes(_infer.encode('utf-8'))
_exploratory = {{'base_notebook_sha256': {BASE_SHA!r}, 'final_epoch_sha256': {EPOCH_SHA!r},
    'adapter_source_sha256': {ADAPTER_SHA!r}, 'original_worker_sha256': SOURCE_SHA['compact_infer.py'],
    'patched_worker_sha256': hashlib.sha256(_infer_path.read_bytes()).hexdigest(),
    'modified_fold_only': 0, 'training_proxy_status': 'REJECTED', 'improvement_established': False,
    'weights_preprocessing_tta_stackers_unchanged': True}}
(WORK / 'exploratory_adapter_receipt.json').write_text(json.dumps(_exploratory, indent=2))
print('EXPLORATORY_ADAPTER_PATCH_READY', json.dumps(_exploratory), flush=True)
'''
    run = ''.join(nb['cells'][7]['source'])
    run = replace_once(run, "'--max-fallback-fraction', '1.0'", "'--max-fallback-fraction', '0.0'")
    nb['cells'][7]['source'] = run.splitlines(True)
    blend = ''.join(nb['cells'][8]['source'])
    blend = replace_once(blend, 'out = np.zeros((n, len(labels)))',
        "assert not r384_fail.any() and not r224_fail.any() and not our_fail.any(), 'exploratory submission forbids reader fallback'\nout = np.zeros((n, len(labels)))")
    nb['cells'][8]['source'] = blend.splitlines(True)
    nb['cells'].insert(7, cell(setup))
    nb['cells'].append(cell('''_worker_logs = list(pred.glob('worker*/log.txt'))
assert _worker_logs and any('EXPLORATORY_ADAPTER_FOLD0_LOADED' in _p.read_text() for _p in _worker_logs)
assert receipt['fallback_studies'] == 0
assert blend['ours_failed'] == blend['raptor384_failed'] == blend['raptor224_failed'] == 0
_exploratory['submission_sha256'] = hashlib.sha256((WORK / 'submission.csv').read_bytes()).hexdigest()
_exploratory['rows'] = len(sub)
_exploratory['zero_fallbacks'] = True
(WORK / 'exploratory_adapter_receipt.json').write_text(json.dumps(_exploratory, indent=2))
print('EXPLORATORY_ADAPTER_COMPLETE', json.dumps(_exploratory), flush=True)
'''))
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    nb['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# Experimental final-epoch rank8 fold0 adaptation\n',
        'User-authorized exploratory submission after failed proxy magnitude gate. No improvement established. '
        'All original attribution and OAI/NDA acknowledgements below retained. Final selections are not changed.\n']})
    target = HERE / 'adapter-exploratory.ipynb'
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')
    metadata = json.loads((base.parent / 'kernel-metadata.json').read_text())
    metadata = {k: metadata[k] for k in ('language', 'kernel_type', 'enable_gpu', 'enable_tpu', 'enable_internet', 'dataset_sources', 'competition_sources', 'model_sources')}
    metadata.update(id='sarthaksharma14/rsna-knee-r6-adapter-exploratory-submission',
        title='RSNA Knee R6 Adapter Exploratory Submission', code_file=target.name,
        is_private=True, kernel_sources=['sarthaksharma14/rsna-knee-r6-lowrank-spatial-adaptation'])
    (HERE / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8', newline='\n')
    receipt = {'base_notebook_sha256': BASE_SHA, 'notebook_sha256': sha(target.read_bytes()),
        'epoch3_sha256': EPOCH_SHA, 'adapter_source_sha256': ADAPTER_SHA,
        'changed_source_cells': ['insert adapter loader after setup', 'zero fallback controller', 'zero fallback blend', 'final adapter receipt'],
        'training_proxy_status': 'REJECTED', 'improvement_established': False,
        'final_selection_changes': False, 'launched': False, 'submitted': False}
    (HERE / 'build_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
    return nb, hook, receipt


if __name__ == '__main__':
    print(json.dumps(build()[2], indent=2))
