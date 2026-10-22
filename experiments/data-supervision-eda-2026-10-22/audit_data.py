"""Fitting-only aggregate EDA. Never parse gold or source-fold0 report groups."""
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / 'r6-own-report-adaptation'))
from report_rules import LABELS, ENTITY, BODY, norm, extract


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fold(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4


def group(report):
    return hashlib.sha256(' '.join(report.casefold().split()).encode()).hexdigest()


def eligible_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[group(row['Report'])].append(row)
    eligible, excluded = [], Counter()
    for key, members in groups.items():
        if any(all(r[label].strip() for label in LABELS) for r in members):
            excluded['gold_or_exact_report_duplicate_groups'] += 1
        elif any(fold(r['StudyInstanceUID']) == 0 for r in members):
            excluded['source_fold0_or_shared_report_groups'] += 1
        else:
            eligible.append((key, members))
    return eligible, dict(excluded)


def summarize(rows, series, parser=extract):
    eligible, excluded = eligible_groups(rows)
    by_uid = defaultdict(list)
    for row in series:
        by_uid[row['StudyInstanceUID']].append(row)
    lengths, series_counts = [], Counter()
    observed, entity_mentions, multientity_mentions = Counter(), Counter(), Counter()
    states = {label: Counter() for label in LABELS}
    metadata_flags, headers, lexical = Counter(), Counter(), Counter()
    for _, members in eligible:
        report = members[0]['Report']
        lengths.append(len(report))
        text = norm(report)
        marker = re.search(BODY, text)
        headers['recognized_findings_heading' if marker else 'no_recognized_heading'] += 1
        # These overlapping lexical indicators are NOT language identification.
        for name, pattern in {'english': r'\b(findings|impression)\b', 'german': r'\b(befund|beurteilung)\b',
                              'spanish': r'\b(resultados|conclusiones|diagnostico)\b',
                              'dutch': r'\b(bevindingen|conclusie)\b', 'french': r'\b(conclusion|resultats)\b'}.items():
            if re.search(pattern, text):
                lexical[name] += 1
        labels = parser(report)
        for label, state in labels.items():
            observed[label] += 1
            states[label][state] += 1
        for clause in re.split(r'[.;\n]+', text[marker.end():] if marker else ''):
            mentions = [label for label, pattern in ENTITY.items() if re.search(pattern, clause)]
            for label in mentions:
                entity_mentions[label] += 1
                if len(mentions) > 1:
                    multientity_mentions[label] += 1
        for member in members:
            records = by_uid[member['StudyInstanceUID']]
            series_counts[str(len(records))] += 1
            if not records:
                metadata_flags['no_series_metadata'] += 1
            planes = {r['Anatomical_Plane'].casefold() for r in records}
            for plane in ('sagittal', 'coronal', 'axial'):
                if plane not in planes:
                    metadata_flags['missing_'+plane] += 1
            fluid = [r for r in records if r['Anatomical_Plane'].casefold() == 'sagittal'
                     and r['Fluid_Sensitive'].strip().casefold() in ('1', 'true', 'yes')]
            if not fluid:
                metadata_flags['no_fluid_sensitive_sagittal_metadata'] += 1
    lengths.sort()
    n = len(eligible)
    return {'fitting_report_groups': n, 'fitting_studies': sum(len(rs) for _, rs in eligible),
        'excluded_groups': excluded, 'parsed_gold_reports': 0, 'parsed_source_fold0_reports': 0,
        'report_length_characters': {str(q): lengths[min(int(q*(len(lengths)-1)),len(lengths)-1)] if lengths else None for q in (.0,.25,.5,.75,1.)},
        'heading_counts': dict(headers), 'overlapping_heading_lexical_indicators': dict(lexical),
        'series_count_histogram': dict(series_counts), 'protocol_metadata_flags': dict(metadata_flags),
        'fluid_sensitive_metadata_values': dict(Counter(r['Fluid_Sensitive'] for r in series)),
        'plane_metadata_values': dict(Counter(r['Anatomical_Plane'] for r in series)),
        'per_finding': {label: {'supported_by_literal_parser': label in ENTITY,
            'known_report_groups': observed[label], 'unknown_report_groups': n-observed[label],
            'coverage': observed[label]/n if n else 0, 'state_counts': dict(states[label]),
            'entity_clauses': entity_mentions[label], 'multi_entity_clauses_rejected': multientity_mentions[label]} for label in LABELS},
        'no_model_or_predictions_read': True, 'new_targets_written': False,
        'scope': 'aggregate descriptive fitting-only EDA; no label correctness or pathology validation'}


def main():
    metadata = ROOT / '.audit/competition-metadata'
    train = metadata / 'train.csv'
    series = metadata / 'train_series.csv'
    assert digest(train) == '8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33'
    assert digest(series) == '573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3'
    with train.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    with series.open(encoding='utf-8-sig', newline='') as f:
        srows = list(csv.DictReader(f))
    result = summarize(rows, srows)
    result['input_sha256'] = {'train': digest(train), 'series': digest(series),
        'frozen_rules': digest(HERE.parent / 'r6-own-report-adaptation/report_rules.py')}
    (HERE / 'aggregate_eda.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
