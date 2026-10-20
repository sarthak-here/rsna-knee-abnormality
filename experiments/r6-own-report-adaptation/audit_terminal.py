"""Read-only exact-v3 terminal audit, never another model-selection screen."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from train import gates, sha, state_losses, write


def same_report(actual, expected):
    """CPU/Kaggle reduction rounding may differ; gate booleans remain exact."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for k in expected:
            same_report(actual[k], expected[k])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            same_report(a, b)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    else:
        assert actual == expected


def prediction_report(path, records, valid):
    with np.load(path, allow_pickle=False) as z:
        expected = {'ids', 'state_targets', 'control_states', 'candidate_states',
                    'control_probabilities', 'candidate_probabilities'}
        assert set(z.files) == expected
        ids = z['ids']
        assert ids.dtype.kind == 'U' and ids.tolist() == [r['id'] for r in records]
        assert len(set(ids.tolist())) == len(records) == 64
        targets = z['state_targets']
        assert targets.dtype == np.int16 and np.array_equal(targets, [r['state_targets'] for r in records])
        for name in ('control_states', 'candidate_states'):
            x = z[name]
            assert x.shape == (64, *valid.shape)
            assert np.isfinite(x).all() and ((0 <= x) & (x <= 1)).all()
            assert np.allclose(x.sum(-1), 1., atol=2e-6)
            assert (x[:, ~valid] == 0).all()
        for name in ('control_probabilities', 'candidate_probabilities'):
            x = z[name]
            assert x.shape == (64, 12) and np.isfinite(x).all() and ((0 <= x) & (x <= 1)).all()
        for j in range(12):
            for t in targets[:, j]:
                assert t == -1 or (0 <= t < valid.shape[1] and valid[j, t])
        result = gates(state_losses(z['control_states'], targets), state_losses(z['candidate_states'], targets),
                       z['control_probabilities'], z['candidate_probabilities'])
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root); own = root/'own_report_adaptation'; out = own/'results'
    here = Path(__file__).resolve().parent
    expected = json.loads((here/'build_receipt.json').read_text())
    assert expected['notebook_sha256'] == '93ee2bf92236862d89736adbd330b4081d00e10da9799ab22b6c4929940b1bbe'
    for name, digest in expected['files_sha256'].items():
        assert sha(own/name) == digest, name
    identity = json.loads((out/'input_identity_receipt.json').read_text())
    old_identity = json.loads((here/'v2_terminal_receipt.json').read_text())['input_identity']
    for key, value in old_identity.items():
        if key != 'own_source_sha256':
            assert identity[key] == value, key
    assert identity['own_source_sha256'] == {name: expected['files_sha256'][name]
                                           for name in ('train.py', 'prepare_splits.py', 'report_rules.py')}
    split = json.loads((own/'splits.json').read_text())
    assert sha(own/'splits.json') == identity['split_sha256']
    report = json.loads((out/'screen_report.json').read_text())
    contract = json.loads((out/'contract.json').read_text())
    assert contract == {'splits_sha256': identity['split_sha256'], 'epochs': 3, 'seed': 20261020,
                        'train_groups': 192, 'tune_groups': 64, 'sealed_groups': 64, 'windows': 32,
                        'lr': 1e-5, 'wd': .01, 'budget_seconds': 10800, 'gold_evaluation': False,
                        'caveat': 'report proxy and conditional image-gradient holdout, NOT whole-pipeline OOF'}
    assert 0 < report['wall_seconds'] <= 10800
    for key in ('gold_evaluated', 'pathology_improvement_established', 'submission_created', 'larger_training_authorized'):
        assert report[key] is False
    cache = json.loads((out/'cache_receipt.json').read_text())
    ids = [r['id'] for k in ('fit','tune','confirmation') for r in split[k]]
    assert [r['id'] for r in cache['entries']] == ids and len(set(ids)) == 320
    assert cache['reader_failures'] == report['reader_failures'] == 0
    assert all(r['feature_shape'] == [32,384,22,22] and r['storage_dtype'] == 'float16' for r in cache['entries'])
    preflight = json.loads((out/'preflight.json').read_text())
    assert preflight['actual_studies'] == 8 and preflight['scaled_objective_gradients_finite']
    assert max(preflight['prefix_full_max_abs_errors']) <= .02
    assert preflight['projected_completion_epoch'] < preflight['deadline_epoch']
    assert sha(args.checkpoint) == identity['checkpoint_sha256']
    reference = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    valid = reference['state_valid'].numpy()
    assert valid.shape[0] == 12
    train_log = json.loads((out/'training_progress.json').read_text())
    assert train_log == report['training'] and len(train_log) == 3
    checkpoint_checks = []
    for epoch, row in enumerate(train_log, 1):
        ckpt = out/f'epoch_{epoch}.pt'
        assert row['epoch'] == epoch and math.isfinite(row['mean_loss'])
        assert sha(ckpt) == row['checkpoint_sha256']
        payload = torch.load(ckpt, map_location='cpu', weights_only=True)
        state = payload['state_dict']
        assert set(state) == set(reference) and payload['epoch'] == epoch
        changed = []
        for name, original in reference.items():
            tensor = state[name]
            assert tensor.shape == original.shape and tensor.dtype == original.dtype
            assert not tensor.is_floating_point() or torch.isfinite(tensor).all()
            if not torch.equal(tensor, original):
                changed.append(name)
        assert len(changed) == 31 and all(n.startswith('enc.stages.3.') for n in changed)
        assert changed == report['changed_tensors']
        states = payload['optimizer']['state']
        assert len(states) == 31
        for entry in states.values():
            assert int(entry['step']) == epoch*192
            assert all(torch.isfinite(t).all() for t in entry.values() if isinstance(t, torch.Tensor))
        assert 1 <= payload['scaler']['scale'] <= 1024
        assert all(k in payload for k in ('torch_rng','cuda_rng','numpy_rng','python_rng'))
        checkpoint_checks.append({'epoch': epoch, 'sha256': row['checkpoint_sha256'],
                                  'optimizer_steps': epoch*192, 'only_final_stage_changed': True})
        del payload, state; gc.collect()
    tuning = prediction_report(out/'tune_predictions.npz', split['tune'], valid)
    same_report(tuning, report['tuning'])
    assert report['tuning'] == json.loads((out/'tuning_report.json').read_text())
    assert report['confirmation_opened'] == tuning['passed']
    confirmation = prediction_report(out/'confirmation_predictions.npz', split['confirmation'], valid) if tuning['passed'] else None
    same_report(confirmation, report['confirmation'])
    if not tuning['passed']:
        assert not (out/'confirmation_predictions.npz').exists()
    expected_status = 'PASSED_PROXY_SCREEN' if confirmation and confirmation['passed'] else 'REJECTED'
    assert report['status'] == expected_status
    retries = json.loads((out/'numerical_retry_receipt.json').read_text())
    assert retries == [{'epoch': 1, 'step': 5, 'retries': [{'scale': 1024., 'loss': 3.1899046897888184,
                                                        'optimizer_step_taken': False}], 'successful_scale': 512.}]
    log = (root/'rsna-knee-r6-own-report-adaptation.log').read_text()
    assert 'OWN_REPORT_ADAPTATION_COMPLETE_NO_SUBMISSION' in log
    assert 'PRIVATE_OWN_REPORT_NOTEBOOK_COMPLETE_NO_SUBMISSION' in log
    assert 'Traceback (most recent call last)' not in log
    assert not list(root.rglob('*.csv'))
    receipt = {'version': 3, 'integrity_passed': True, 'scientific_status': expected_status,
               'metric_reproduction_tolerance': 1e-12, 'gate_booleans_reproduced_exactly': True,
               'checkpoint_checks': checkpoint_checks, 'optimizer_steps': 576,
               'temporary_cache_bytes_rehashed': False, 'cached_studies_per_receipt': 320,
               'reader_failures_per_receipt': 0, 'numerical_retries': retries,
               'log_warning_free': False, 'no_traceback': True, 'no_csv_or_submission': True,
               'tuning': tuning, 'confirmation': confirmation, 'wall_seconds': report['wall_seconds'],
               'source_identity': identity,
               'downloaded_file_sha256': {str(x.relative_to(root)): sha(x) for x in root.rglob('*') if x.is_file()}}
    write(args.out, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('downloaded_file_sha256','source_identity')}, indent=2))


if __name__ == '__main__':
    main()
