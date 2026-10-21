"""Read-only terminal validation of the frozen rank8 pilot; no model selection."""
import argparse
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from train import sha, write

# Reuse pickle-free prediction and exact metric/gate comparison helpers.
helper_path = Path(__file__).resolve().parent.parent / 'r6-own-report-adaptation/audit_terminal.py'
spec = importlib.util.spec_from_file_location('prior_terminal_helpers', helper_path)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
prediction_report, same_report = helpers.prediction_report, helpers.same_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    own = root / 'r6_lowrank_adaptation'
    out = own / 'results'
    here = Path(__file__).resolve().parent
    build = json.loads((here / 'build_receipt.json').read_text())
    assert build['notebook_sha256'] == '6c94ff834fcf66ef7ac7d495ce575dd7693e27c21fb36dbecebfb9a3131ba19b'
    for name, digest in build['files_sha256'].items():
        assert sha(own / name) == digest, name
    identity = json.loads((out / 'input_identity_receipt.json').read_text())
    prior = json.loads((here.parent / 'r6-own-report-adaptation/v3_terminal_receipt.json').read_text())['source_identity']
    for key, value in prior.items():
        if key not in ('split_sha256', 'own_source_sha256'):
            assert identity[key] == value, key
    assert identity['split_sha256'] == build['split_sha256']
    assert identity['own_source_sha256'] == {n: build['files_sha256'][n] for n in
        ('train.py', 'prepare_splits.py', 'report_rules.py', 'adapters.py', 'excluded_previous_split.json')}
    split = json.loads((own / 'splits.json').read_text())
    # Exact split bytes were pinned before launch, including exclusion guards.
    assert sha(own / 'splits.json') == build['split_sha256']
    contract = json.loads((out / 'contract.json').read_text())
    assert contract == {'splits_sha256': build['split_sha256'], 'epochs': 3, 'seed': 20261022,
        'train_groups': 192, 'tune_groups': 64, 'sealed_groups': 64, 'windows': 32,
        'lr': 1e-5, 'wd': .01, 'budget_seconds': 10800, 'gold_evaluation': False,
        'architecture': 'final_stage_fc2_rank8', 'adapter_rank': 8, 'trainable_parameters': 92160,
        'caveat': 'report proxy and conditional image-gradient holdout, NOT whole-pipeline OOF'}
    report = json.loads((out / 'screen_report.json').read_text())
    assert 0 < report['wall_seconds'] <= 10800
    for key in ('gold_evaluated', 'pathology_improvement_established', 'submission_created', 'larger_training_authorized'):
        assert report[key] is False
    assert report['original_tensors_unchanged'] and report['adapter_parameters'] == 92160
    cache = json.loads((out / 'cache_receipt.json').read_text())
    ids = [r['id'] for part in ('fit', 'tune', 'confirmation') for r in split[part]]
    assert len(set(ids)) == len(ids) == 320
    assert [r['id'] for r in cache['entries']] == ids
    assert cache['reader_failures'] == report['reader_failures'] == 0
    assert all(r['feature_shape'] == [32, 384, 22, 22] and r['storage_dtype'] == 'float16' for r in cache['entries'])
    preflight = json.loads((out / 'preflight.json').read_text())
    assert preflight['actual_studies'] == 8 and preflight['scaled_objective_gradients_finite']
    assert max(preflight['prefix_full_max_abs_errors']) <= .02
    assert preflight['projected_completion_epoch'] < preflight['deadline_epoch']
    assert sha(args.checkpoint) == identity['checkpoint_sha256']
    reference = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    valid = reference['state_valid'].numpy()
    training = json.loads((out / 'training_progress.json').read_text())
    assert training == report['training'] and len(training) == 3
    checks = []
    expected_adapters = {f'enc.stages.3.blocks.{b}.mlp.fc2.adapter_{d}.weight': shape
        for b in range(3) for d, shape in [('down', (8, 3072)), ('up', (768, 8))]}
    for epoch, row in enumerate(training, 1):
        path = out / f'epoch_{epoch}.pt'
        assert row['epoch'] == epoch and math.isfinite(row['mean_loss'])
        assert sha(path) == row['checkpoint_sha256']
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert payload['epoch'] == epoch and payload['adapter_rank'] == 8
        assert payload['initial_checkpoint_sha256'] == identity['checkpoint_sha256']
        state = payload['state_dict']
        original_keys = set()
        for name, tensor in reference.items():
            key = name.replace('.mlp.fc2.', '.mlp.fc2.base.') if name.startswith('enc.stages.3.blocks.') else name
            assert key in state and state[key].dtype == tensor.dtype and torch.equal(tensor, state[key]), name
            original_keys.add(key)
        assert set(state) - original_keys == set(expected_adapters)
        for name, shape in expected_adapters.items():
            assert tuple(state[name].shape) == shape and torch.isfinite(state[name]).all()
        states = payload['optimizer']['state']
        assert len(states) == 6
        for item in states.values():
            assert int(item['step']) == epoch * 192
            assert all(torch.isfinite(t).all() for t in item.values() if isinstance(t, torch.Tensor))
        assert 1 <= payload['scaler']['scale'] <= 1024
        assert all(k in payload for k in ('torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng'))
        checks.append({'epoch': epoch, 'sha256': row['checkpoint_sha256'],
            'optimizer_steps': epoch * 192, 'original_tensors_unchanged': True, 'finite_adapter_tensors': 6})
        del payload, state
        gc.collect()
    assert set(report['changed_tensors']) <= set(expected_adapters) and report['changed_tensors']
    tuning = prediction_report(out / 'tune_predictions.npz', split['tune'], valid)
    same_report(tuning, report['tuning'])
    same_report(tuning, json.loads((out / 'tuning_report.json').read_text()))
    assert report['confirmation_opened'] == tuning['passed']
    confirmation = prediction_report(out / 'confirmation_predictions.npz', split['confirmation'], valid) if tuning['passed'] else None
    same_report(confirmation, report['confirmation'])
    if not tuning['passed']:
        assert not (out / 'confirmation_predictions.npz').exists()
    status = 'PASSED_PROXY_SCREEN' if confirmation and confirmation['passed'] else 'REJECTED'
    assert report['status'] == status
    retry_path = out / 'numerical_retry_receipt.json'
    retries = json.loads(retry_path.read_text()) if retry_path.exists() else []
    for event in retries:
        assert 1 <= event['epoch'] <= 3 and 1 <= event['step'] <= 192
        scales = [r['scale'] for r in event['retries']] + [event['successful_scale']]
        assert all(1 <= x <= 1024 for x in scales)
        assert all(b == a / 2 for a, b in zip(scales, scales[1:]))
        assert all(not r['optimizer_step_taken'] and math.isfinite(r['loss']) for r in event['retries'])
    log = (root / 'rsna-knee-r6-lowrank-spatial-adaptation.log').read_text()
    assert 'R6_LOWRANK_ADAPTATION_COMPLETE_NO_SUBMISSION' in log
    assert 'PRIVATE_R6_LOWRANK_NOTEBOOK_COMPLETE_NO_SUBMISSION' in log
    assert 'Traceback (most recent call last)' not in log
    assert not list(root.rglob('*.csv'))
    receipt = {'version': 1, 'integrity_passed': True, 'scientific_status': status,
        'checkpoint_checks': checks, 'optimizer_steps': 576, 'tuning': tuning, 'confirmation': confirmation,
        'wall_seconds': report['wall_seconds'], 'reader_failures_per_receipt': 0,
        'temporary_cache_bytes_rehashed': False, 'numerical_retries': retries,
        'original_tensors_unchanged': True, 'no_csv_or_submission': True, 'no_traceback': True,
        'log_warning_free': False, 'source_identity': identity,
        'downloaded_file_sha256': {str(x.relative_to(root)): sha(x) for x in root.rglob('*') if x.is_file()}}
    write(args.out, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('source_identity', 'downloaded_file_sha256')}, indent=2))


if __name__ == '__main__':
    main()
