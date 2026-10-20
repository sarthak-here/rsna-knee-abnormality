"""Freeze report groups and own state targets without parsing organizer gold."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from report_rules import LABELS, VERSION, extract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(train_path, spec_path):
    if sha(train_path) != "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33":
        raise ValueError("official metadata identity drift")
    if sha(spec_path) != "3914242b0c404aee625ea65801f14f60adfee326c1ddd6fb96c48ed7050b7ff3":
        raise ValueError("source state specification drift")
    with Path(train_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 4407 or len({r["StudyInstanceUID"] for r in rows}) != 4407:
        raise ValueError("official population mismatch")
    spec = json.loads(Path(spec_path).read_text())
    groups = defaultdict(list)
    gold = set()
    fold = {}
    for row in rows:
        uid = row["StudyInstanceUID"]
        group = hashlib.sha256(" ".join(row["Report"].casefold().split()).encode()).hexdigest()
        groups[group].append(row)
        fold[uid] = int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4
        if all(row[l].strip() for l in LABELS):
            gold.add(uid)
    if len(gold) != 58:
        raise ValueError("gold exclusion count mismatch")
    excluded = {g for g, rs in groups.items() if any(r["StudyInstanceUID"] in gold for r in rs)}
    fit, hold = [], []
    counts = {"gold": len(gold), "gold_report_duplicate_non_gold": sum(len(groups[g]) for g in excluded) - len(gold),
              "eligible_source_fit_studies": 0, "eligible_guarded_holdout_studies": 0, "parsed_gold_reports": 0}
    for group, members in groups.items():
        if group in excluded:
            continue
        outside = [r for r in members if fold[r["StudyInstanceUID"]] != 0]
        guarded = all(fold[r["StudyInstanceUID"]] == 0 for r in members)
        counts["eligible_source_fit_studies"] += len(outside)
        counts["eligible_guarded_holdout_studies"] += len(members) if guarded else 0
        states = extract(members[0]["Report"])
        if not states:
            continue
        target = [spec["categories"][l].index(states[l]) if l in states else -1 for l in LABELS]
        candidates = outside if outside else members
        uid = min(r["StudyInstanceUID"] for r in candidates)
        record = {"id": uid, "report_group": group, "state_targets": target}
        if outside:
            fit.append(record)
        elif guarded:
            hold.append(record)
    if counts["eligible_source_fit_studies"] != 3317 or counts["eligible_guarded_holdout_studies"] != 977:
        raise ValueError("source exclusion contract drift")
    order = lambda prefix, x: hashlib.sha256((prefix + x["report_group"]).encode()).hexdigest()
    fit.sort(key=lambda x: order("fit_v1", x))
    hold.sort(key=lambda x: order("hold_v1", x))
    result = {"version": VERSION, "metadata_sha256": sha(train_path), "state_spec_sha256": sha(spec_path),
              "rules_sha256": sha(Path(__file__).with_name("report_rules.py")), "exclusion_counts": counts,
              "covered_fit_groups": len(fit), "covered_guarded_holdout_groups": len(hold),
              "status": "READY" if len(fit) >= 192 and len(hold) >= 128 else "COVERAGE_REJECTION",
              "fit": fit[:192], "tune": hold[:64], "confirmation": hold[64:128],
              "pathology_truth": False, "submission_created": False}
    all_records = result["fit"] + result["tune"] + result["confirmation"]
    if len({r["id"] for r in all_records}) != len(all_records) or len({r["report_group"] for r in all_records}) != len(all_records):
        raise ValueError("partition overlap")
    result["state_observation_counts"] = {
        partition: {l: sum(r["state_targets"][j] >= 0 for r in result[partition]) for j, l in enumerate(LABELS)}
        for partition in ("fit", "tune", "confirmation")}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--spec", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    result = prepare(a.train, a.spec)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("fit", "tune", "confirmation")}, indent=2))


if __name__ == "__main__":
    main()
