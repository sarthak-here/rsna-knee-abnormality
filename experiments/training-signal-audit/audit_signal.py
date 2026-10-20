"""CPU fitting-label leverage audit. No predictions, model, GPU or holdout loss."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'r6-own-report-adaptation'))
from report_rules import LABELS, extract


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(metadata, spec_path, split_path):
    assert digest(metadata) == '8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33'
    assert digest(spec_path) == '3914242b0c404aee625ea65801f14f60adfee326c1ddd6fb96c48ed7050b7ff3'
    split = json.loads(Path(split_path).read_text())
    spec = json.loads(Path(spec_path).read_text())
    with Path(metadata).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4407
    by_id = {r['StudyInstanceUID']: r for r in rows}
    group = lambda r: hashlib.sha256(' '.join(r['Report'].casefold().split()).encode()).hexdigest()
    fold = lambda r: int(hashlib.sha1(r['StudyInstanceUID'].encode()).hexdigest()[:8], 16) % 4
    gold_ids = {r['StudyInstanceUID'] for r in rows if all(r[l].strip() for l in LABELS)}
    excluded_groups = {group(by_id[uid]) for uid in gold_ids}
    assert len(gold_ids) == 58
    # Inspect only existing fitting supervision, never tuning/confirmation reports.
    fit = split['fit']; assert len(fit) == 192
    categories = {l: Counter() for l in LABELS}
    for record in fit:
        row = by_id[record['id']]
        assert record['id'] not in gold_ids and group(row) not in excluded_groups and fold(row) != 0
        states = extract(row['Report'])
        recreated = [spec['categories'][l].index(states[l]) if l in states else -1 for l in LABELS]
        assert recreated == record['state_targets']
        for label, state in states.items():
            categories[label][state] += 1
    counts = np.array([sum(categories[l].values()) for l in LABELS])
    active = int((counts > 0).sum())
    signal = {}
    for label, count in zip(LABELS, counts):
        count = int(count)
        coefficient = 192./(active*count) if count else 0.
        signal[label] = {'known_fit_cells': count, 'fit_coverage': count/192,
                         'state_counts': dict(categories[label]),
                         'coefficient_per_known_ce_cell': coefficient,
                         'epoch_sum_ce_coefficients': count*coefficient,
                         'direct_report_supervision': bool(count)}
    # Reserve scope is based only on source fold/group membership and prior
    # pilot membership. Do not extract unused reports or score their labels.
    grouped = defaultdict(list)
    for row in rows:
        grouped[group(row)].append(row)
    consumed = {r['report_group'] for part in ('fit','tune','confirmation') for r in split[part]}
    remaining = sorted(g for g, members in grouped.items()
                       if g not in excluded_groups and g not in consumed and all(fold(r) == 0 for r in members))
    return {'kind': 'FITTING_SUPERVISION_AND_PILOT_EXPOSURE_AUDIT',
            'metadata_sha256': digest(metadata), 'spec_sha256': digest(spec_path),
            'normalized_split_sha256': hashlib.sha256(Path(split_path).read_text().encode()).hexdigest(),
            'fit_groups': 192, 'known_fit_cells': int(counts.sum()), 'active_supervised_classes': active,
            'per_class': signal,
            'max_to_min_positive_cell_coefficient': float(counts.max()/counts[counts > 0].min()),
            'parser_calls': 192, 'gold_report_parser_calls': 0, 'tune_confirmation_report_parser_calls': 0,
            'remaining_guarded_source_fold0_groups_not_in_pilot': len(remaining),
            'remaining_group_digest': hashlib.sha256('\n'.join(remaining).encode()).hexdigest(),
            'remaining_group_labels_inspected': False,
            'globally_fresh_or_independent_evaluation_proven': False,
            'scope_caveat': 'not used in this pilot is not proof of exclusion from all prior research/source priors',
            'model_or_prediction_loaded': False, 'gpu_used': False, 'candidate_authorized': False,
            'closed_contract_changed': False, 'causal_reason_for_failure_proven': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', type=Path, default=REPO/'.audit/competition-metadata/train.csv')
    p.add_argument('--spec', type=Path, default=REPO/'.audit/r6-strict-load/state_spec.json')
    p.add_argument('--split', type=Path, default=HERE.parent/'r6-own-report-adaptation/splits.json')
    p.add_argument('--out', type=Path, default=HERE/'signal_receipt.json')
    a = p.parse_args()
    result = audit(a.metadata, a.spec, a.split)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
