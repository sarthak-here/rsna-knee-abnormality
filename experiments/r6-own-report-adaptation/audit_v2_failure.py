"""Strict identity and terminal receipt audit for the exact failed v2 output."""
import hashlib
import json
from pathlib import Path
from train import sha, write


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]/'.audit/own-report-v2'
    own = root/'own_report_adaptation'; results = own/'results'
    identity = json.loads((results/'input_identity_receipt.json').read_text())
    assert identity['own_source_sha256']['train.py'] == '4cdcea1b9e6210f9c0b5ddedb31f6eb506b9c26c9002fe6077c0c4822016da00'
    for name in ('train.py', 'report_rules.py', 'prepare_splits.py'):
        assert sha(own/name) == identity['own_source_sha256'][name]
    assert sha(own/'splits.json') == hashlib.sha256((here/'splits.json').read_text().encode()).hexdigest() == identity['split_sha256']
    split = json.loads((own/'splits.json').read_text())
    ids = [r['id'] for k in ('fit', 'tune', 'confirmation') for r in split[k]]
    cache = json.loads((results/'cache_receipt.json').read_text())
    assert [r['id'] for r in cache['entries']] == ids and len(set(ids)) == 320
    assert cache['reader_failures'] == 0
    assert all(r['feature_shape'] == [32,384,22,22] and r['storage_dtype'] == 'float16' for r in cache['entries'])
    failure = json.loads((results/'runtime_failure.json').read_text())
    assert failure['epoch'] == 1 and failure['optimizer_steps'] == failure['completed_steps_this_epoch'] == 4
    assert failure['amp_scale'] == 1024 and failure['finite_objective'] == 3.1899046897888184
    preflight = json.loads((results/'preflight.json').read_text())
    assert preflight['scaled_objective_gradients_finite'] and preflight['amp_initial_scale'] == 1024
    assert max(preflight['prefix_full_max_abs_errors']) <= .02
    assert preflight['projected_completion_epoch'] < preflight['deadline_epoch']
    assert not list(root.rglob('*.csv')) and not list(root.rglob('epoch_*.pt'))
    assert not (results/'tuning_report.json').exists() and not (results/'screen_report.json').exists()
    report = {'version': 2, 'status': 'RUNTIME_FAILURE_NOT_SCIENTIFIC_REJECTION',
              'failure': failure, 'cached_studies': 320, 'reader_failures': 0,
              'no_epoch_checkpoint': True, 'no_evaluation_report': True,
              'accuracy_result': None, 'temporary_cache_bytes_available': False,
              'no_retained_csv': True, 'no_submission': True,
              'preflight': preflight, 'input_identity': identity,
              'downloaded_file_sha256': {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}}
    write(here/'v2_terminal_receipt.json', report)
    print(json.dumps({'status': report['status'], 'failure': failure, 'cached_studies': 320}))


if __name__ == '__main__':
    main()
