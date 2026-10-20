"""Trace released R6 target identities and reject false reproduction claims.

This reads metadata only. It neither creates replacement targets nor approves
training, and it does not infer pathology quality from schema compatibility.
"""
import csv
import hashlib
import json
from pathlib import Path
import re


def digest(path, algorithm="sha256"):
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def inspect_table(path, labels):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    required = ["StudyInstanceUID", "is_gold", "review_holdout", "fold"]
    required += [f"{label}|{suffix}" for label in labels for suffix in ("strict", "prior", "lr_prior", "w")]
    result = {"file": str(path), "sha256": digest(path), "md5": digest(path, "md5"),
              "rows": len(rows), "missing_original_lr_columns": [x for x in required if x not in columns]}
    if all(x in columns for x in ("is_gold", "review_holdout", "fold")) and all(f"{l}|prior" in columns for l in labels):
        non_gold = [r for r in rows if r["is_gold"] == "0"]
        blanked = [r for r in non_gold if any(r[f"{l}|prior"] == "" for l in labels)]
        pool = [r for r in non_gold if all(r[f"{l}|prior"] != "" for l in labels)
                and r["review_holdout"] == "0" and int(r["fold"]) != -3]
        result.update(non_gold_rows=len(non_gold), blanked_prior_rows=len(blanked),
                      source_default_pool_from_this_table=len(pool),
                      pool_matches_released_4344=len(pool) == 4344)
    return result


def main():
    repo = Path(__file__).resolve().parents[2]
    spec_path = repo / ".audit/r6-strict-load/state_spec.json"
    trainer_path = repo / ".audit/rsna-knee-blend-code/r6src/compact_train.py"
    if digest(spec_path) != "3914242b0c404aee625ea65801f14f60adfee326c1ddd6fb96c48ed7050b7ff3":
        raise ValueError("released state specification drift")
    if digest(trainer_path) != "04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba":
        raise ValueError("released trainer drift")
    spec = json.loads(spec_path.read_text())
    identities = [{"filename": name, "md5": md5} for name, md5 in
                  re.findall(r"([\w.]+) \(([0-9a-f]{32})\)", spec["source_read"])]
    if len(identities) != 3:
        raise ValueError("unexpected original supervision lineage format")
    paths = [repo / ".audit/labels-soft-v2/labels_soft_v2_targets.csv",
             repo / ".audit/labels-soft-v2/labels_states_v2_shard0.csv",
             repo / ".audit/newest/haung-v2/weak_labels_llm_v2.csv"]
    tables = [inspect_table(p, spec["labels"]) for p in paths]
    original_hashes = {r["md5"] for r in identities}
    for table in tables:
        table["matches_any_embedded_original_md5"] = table["md5"] in original_hashes
        table["file"] = str(Path(table["file"]).relative_to(repo))
    other_pool = repo / ".audit/bleg-cpu-audit/pool_uids.json"
    other_config = repo / ".audit/bleg-cpu-audit/run_config.json"
    other_rows = json.loads(other_pool.read_text())
    other_run = json.loads(other_config.read_text())
    report = {
        "scope": "original target lineage and local substitute schema audit, not an accuracy screen",
        "state_spec_sha256": digest(spec_path), "trainer_sha256": digest(trainer_path),
        "source_read": spec["source_read"], "embedded_original_identities": identities,
        "deployed_training_target_filename": "probe_targets_opusv11b_rr6_plr6_cdr.csv",
        "embedded_parent_is_not_final_cdr_target": True,
        "local_alternatives": tables,
        "important_objective_detail": "lr_loss applies torch.log(lr): lr_prior is a positive likelihood ratio, not a raw probability or an already-log ratio",
        "pool_difference_status": "unresolved: trainer excludes blanked priors, review holdout and fold -3, and requires rendered cache availability; five IDs cannot be inferred from counts alone",
        "other_local_pool": {"path": ".audit/bleg-cpu-audit/pool_uids.json", "rows": len(other_rows),
                             "sha256": digest(other_pool), "config_sha256": digest(other_config),
                             "arm": other_run["arms"], "target_path": other_run["targets"],
                             "must_not_substitute_for_R6_pool": True},
        "original_supervision_recovered": False, "reproduction_ready": False,
        "alternate_objective_proven_invalid": False,
        "next_gate": "Recover authenticated original tables/pool, or explicitly justify a different objective with frozen heads/state semantics, valid holdouts and a measured bounded paired test; no automatic GPU launch",
        "gpu_used": False, "optimizer_steps": 0, "model_accuracy_gain_established": False,
    }
    out = Path(__file__).with_name("supervision_receipt.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
