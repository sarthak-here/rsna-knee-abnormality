"""New candidate groups: exclude gold and EVERY prior pilot selected group."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from report_rules import LABELS, extract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(train_path, spec_path):
    if sha(train_path) != "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33":
        raise ValueError("official metadata changed")
    if sha(spec_path) != "3914242b0c404aee625ea65801f14f60adfee326c1ddd6fb96c48ed7050b7ff3":
        raise ValueError("state specification changed")
    if sha(Path(__file__).with_name("report_rules.py")) != "e10ed7ed45841ab058ffe6926e8bddf01e46a67da2fcf149c6dae0f49569b736":
        raise ValueError("frozen report rules changed")
    previous = json.loads(Path(__file__).with_name("excluded_previous_split.json").read_text())
    excluded_previous = set(previous["report_groups"])
    if len(excluded_previous) != 320:
        raise ValueError("prior-pilot exclusion mismatch")
    with Path(train_path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 4407 or len({r["StudyInstanceUID"] for r in rows}) != 4407:
        raise ValueError("population mismatch")
    groups = defaultdict(list)
    gold = set()
    for row in rows:
        group = hashlib.sha256(" ".join(row["Report"].casefold().split()).encode()).hexdigest()
        groups[group].append(row)
        if all(row[l].strip() for l in LABELS):
            gold.add(row["StudyInstanceUID"])
    if len(gold) != 58:
        raise ValueError("gold exclusion mismatch")
    gold_groups = {g for g, rs in groups.items() if any(r["StudyInstanceUID"] in gold for r in rs)}
    spec = json.loads(Path(spec_path).read_text())
    fit, hold = [], []
    order = sorted(groups, key=lambda g: hashlib.sha256(("rank8_20261022:" + g).encode()).hexdigest())
    fold = lambda uid: int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4
    for g in order:
        if g in gold_groups or g in excluded_previous:
            continue
        members = groups[g]
        outside = [r for r in members if fold(r["StudyInstanceUID"]) != 0]
        partition = fit if outside else hold
        limit = 192 if outside else 128
        if len(partition) >= limit:
            continue
        # Gold/old groups are excluded BEFORE any parser call.
        states = extract(members[0]["Report"])
        if not states:
            continue
        targets = [spec["categories"][l].index(states[l]) if l in states else -1 for l in LABELS]
        uid = min(r["StudyInstanceUID"] for r in (outside if outside else members))
        partition.append({"id": uid, "report_group": g, "state_targets": targets})
    result = {"version": "rank8_spatial_v1", "status": "READY" if len(fit) == 192 and len(hold) == 128 else "COVERAGE_REJECTION",
              "metadata_sha256": sha(train_path), "rules_sha256": sha(Path(__file__).with_name("report_rules.py")),
              "previous_split_sha256": previous["source_sha256"], "previous_groups_excluded": len(excluded_previous),
              "gold_excluded": 58, "parsed_gold_reports": 0,
              "fit": fit, "tune": hold[:64], "confirmation": hold[64:128],
              "evaluation_scope": "new pilot-unselected report groups; source fold0 additional-gradient holdout, NOT full-pipeline OOF or independent pathology truth",
              "gold_evaluation": False, "submission_created": False}
    records = fit + hold
    if len({r["report_group"] for r in records}) != len(records) or any(r["report_group"] in excluded_previous for r in records):
        raise ValueError("new/old partition overlap")
    if any(fold(r["id"]) != 0 for r in hold) or any(fold(r["id"]) == 0 for r in fit):
        raise ValueError("source-fold exclusion violation")
    result["fit_state_counts"] = [sum(r["state_targets"][j] >= 0 for r in fit) for j in range(12)]
    return result
