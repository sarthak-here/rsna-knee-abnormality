"""CPU-only image-gradient holdout audit; not whole-pipeline OOF certification."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def digest(path, kind="sha256"):
    return hashlib.new(kind, Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trainer", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if digest(args.trainer) != manifest["src_sha256"]["compact_train.py"]:
        raise ValueError("trainer SHA differs from published manifest")
    if digest(args.trainer, "md5") != manifest["trainer_md5"]:
        raise ValueError("trainer MD5 differs from published manifest")
    if manifest["run_config"]["folds"] != 4:
        raise ValueError("unexpected source folds")
    source = Path(args.trainer).read_text(encoding="utf-8")
    for required in ("int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K",
                     "else np.flatnonzero(pool_fold != k)"):
        if required not in source:
            raise ValueError("review fold-rule source manually before use")
    with open(args.train, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 4407 or len({r["StudyInstanceUID"] for r in rows}) != 4407:
        raise ValueError("official population drift")
    groups = defaultdict(list)
    gold = set()
    folds = {}
    for row in rows:
        uid = row["StudyInstanceUID"]
        report = " ".join(row["Report"].casefold().split())
        if not report:
            raise ValueError("blank report group")
        groups[hashlib.sha256(report.encode()).hexdigest()].append(uid)
        if all(row[t].strip() for t in TARGETS):
            gold.add(uid)
        folds[uid] = int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4
    if len(gold) != 58:
        raise ValueError("official gold count drift")
    gold_reports = {g for g, members in groups.items() if set(members) & gold}
    eligible = {u for g, members in groups.items() if g not in gold_reports for u in members}
    result = []
    for k in range(4):
        validation = {u for g, members in groups.items()
                      if g not in gold_reports and all(folds[u] == k for u in members)
                      for u in members}
        fitting = {u for u in eligible if folds[u] != k}
        result.append({"fold": k, "eligible_non_gold_outside_source_fold": len(fitting),
                       "hash_holdout_before_report_guard": sum(folds[u] == k for u in eligible),
                       "report_guarded_image_holdout": len(validation),
                       "holdout_ids_sha256": hashlib.sha256("\n".join(sorted(validation)).encode()).hexdigest()})
    print(json.dumps({"source_matches_manifest_sha_and_md5": True,
                      "manifest_pool_count": manifest["run_config"]["pool_studies"],
                      "official_non_gold": 4349,
                      "gold_report_duplicate_rows_excluded": 4349 - len(eligible),
                      "fold_rule": "int(sha1(uid).hexdigest()[:8],16)%4",
                      "folds": result,
                      "scope": "conditional image-gradient exclusion, not independent pathology truth or full-pipeline OOF",
                      "caveats": ["Published manifest is the source of training-lineage claims.",
                                  "Original trainer derives class bias from the full pool.",
                                  "State priors/target generation/postprocessors require separate lineage audit.",
                                  "No checkpoint loaded, labels scored, or GPU used."]}, indent=2))


if __name__ == "__main__":
    main()
