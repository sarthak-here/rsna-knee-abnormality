"""Read-only all-ref Git blob audit; prints paths/counts, never medical values.

This is a targeted release check, not a comprehensive security certification.
Run from the repository root. Optional official metadata stays local.
"""
import csv
import json
import pathlib
import re
import subprocess


def main():
    metadata = pathlib.Path('.audit/competition-metadata/train.csv')
    ids = set()
    if metadata.exists():
        with metadata.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key, value in row.items():
                    if key and ('uid' in key.lower() or 'study' in key.lower()):
                        if value and (re.fullmatch(r'[0-9a-f]{12,64}', value) or re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', value)):
                            ids.add(value.encode())
    paths = {}
    for line in subprocess.check_output(['git', 'rev-list', '--objects', '--all'], text=True).splitlines():
        oid, _, path = line.partition(' ')
        paths.setdefault(oid, path)
    proc = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    findings = []
    secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    token = re.compile(rb'(?<![0-9a-f.])(?:[0-9]+(?:\.[0-9]+)+|[0-9a-f]{12,64})(?![0-9a-f.])')
    blob_count = 0
    for oid, path in paths.items():
        proc.stdin.write((oid + '\n').encode())
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        if header[1] != b'blob':
            continue
        blob_count += 1
        reasons = []
        count = len(set(token.findall(data)) & ids)
        if count:
            reasons.append('official_study_identifiers')
        if secret.search(data):
            reasons.append('credential_pattern_requires_review')
        if path.endswith('splits.json') and b'state_targets' in data:
            reasons.append('per_study_supervision_split')
        if pathlib.PurePosixPath(path).suffix in {'.csv', '.npz', '.parquet', '.dcm', '.pt', '.pth', '.ckpt'}:
            reasons.append('data_or_weight_artifact_requires_review')
        if reasons:
            findings.append({'path': path, 'blob_sha': oid, 'bytes': size, 'reasons': reasons, 'distinct_official_ids': count})
    proc.stdin.close()
    proc.wait()
    print(json.dumps({'scope': 'all locally reachable refs; unique Git blobs', 'blobs_scanned': blob_count,
                      'official_metadata_available': metadata.exists(), 'known_official_id_count': len(ids),
                      'findings': findings, 'public_release_approved': False,
                      'limitations': 'Exact known-ID and selected secret patterns only; absence of hits is not proof of safe content. No artifact rights or full medical free-text review.'}, indent=2))


if __name__ == '__main__':
    main()
