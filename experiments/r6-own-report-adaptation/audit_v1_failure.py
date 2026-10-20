"""Verify preserved v1 runtime evidence, not an accuracy/promotion test."""
import json
import hashlib
from pathlib import Path
from train import sha, write


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]/'.audit/own-report-v1'
    own = root/'own_report_adaptation'; results = own/'results'
    identity = json.loads((results/'input_identity_receipt.json').read_text())
    assert identity['own_source_sha256']['train.py'] == 'ef77eb3ae3012f55342ced3c6326dae1ca0c04139874f6f9ae1f63d21dab1424'
    for name in ('train.py', 'report_rules.py', 'prepare_splits.py'):
        assert sha(own/name) == identity['own_source_sha256'][name]
    # Notebook embeds read_text() content (LF); Git checkout may have CRLF.
    local_split_sha = hashlib.sha256((here/'splits.json').read_text().encode()).hexdigest()
    assert sha(own/'splits.json') == local_split_sha == identity['split_sha256']
    split = json.loads((own/'splits.json').read_text())
    ids = [r['id'] for k in ('fit', 'tune', 'confirmation') for r in split[k]]
    cache = json.loads((results/'cache_receipt.json').read_text())
    entries = cache['entries']
    assert len(entries) == len(set(ids)) == 320 and [r['id'] for r in entries] == ids
    assert cache['reader_failures'] == 0
    assert all(r['feature_shape'] == [32, 384, 22, 22] and r['storage_dtype'] == 'float16' for r in entries)
    preflight = json.loads((results/'preflight.json').read_text())
    assert preflight['actual_studies'] == 8 and max(preflight['prefix_full_max_abs_errors']) <= .02
    assert preflight['projected_completion_epoch'] < preflight['deadline_epoch']
    log = (root/'rsna-knee-r6-own-report-adaptation.log').read_text()
    assert 'nonfinite gradient, refusing silent scaler-skipped step' in log
    assert not list(root.rglob('*.csv')) and not list(root.rglob('epoch_*.pt'))
    assert not (results/'tuning_report.json').exists() and not (results/'screen_report.json').exists()
    report = {'version': 1, 'status': 'RUNTIME_FAILURE_NOT_SCIENTIFIC_REJECTION',
              'failure': 'finite objective followed by nonfinite gradient norm during AMP training',
              'cached_studies': 320, 'reader_failures': 0, 'completed_epoch_checkpoints': 0,
              'completed_optimizer_steps': None, 'step_count_caveat': 'v1 did not persist per-step progress before failure',
              'tuning_or_confirmation_report': False, 'accuracy_result': None,
              'cache_files_available_for_local_rehash': False,
              'cache_caveat': 'temporary GPU feature files not published; receipt identities checked, not downloaded pixel/feature content',
              'no_retained_csv': True, 'no_submission': True,
              'preflight': preflight, 'input_identity': identity,
              'downloaded_file_sha256': {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}}
    write(here/'v1_terminal_receipt.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('downloaded_file_sha256', 'input_identity', 'preflight')}, indent=2))


if __name__ == '__main__':
    main()
