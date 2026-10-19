"""Read-only terminal validation; no tuning, inference, or gold scoring."""
import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from search import groups_from_reports, partition, TARGETS


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root); base = root / "transfer_optuna"; output = base / "search-output"
    contract = json.loads((output / "contract.json").read_text())
    report = json.loads((output / "screen_report.json").read_text())
    assets = json.loads((base / "weights/assets.json").read_text())
    preflight = json.loads((base / "preflight.json").read_text())
    assert sha(base / "search.py") == contract["source_sha256"]
    assert sha(base / "models.py") == contract["models_source_sha256"]
    assert sha(base / "weights/assets.json") == contract["assets_sha256"]
    assert sha(base / "preflight.json") == assets["preflight"]["sha256"]
    assert sha(args.train) == assets["train"]["sha256"]
    assert sha(args.labels) == assets["labels"]["sha256"]
    assert preflight["models_source_sha256"] == contract["models_source_sha256"]
    assert all(r["passed"] for r in preflight["models"])
    assert preflight["weight_hashes"] == {k: assets[k]["sha256"] for k in ("swin3d_t", "videomae")}
    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    ids = train.StudyInstanceUID.to_numpy(); gold = train[TARGETS].notna().all(axis=1).to_numpy()
    expected = partition(ids, groups_from_reports(train.Report.tolist()), gold)
    assert {k: ids[v].tolist() for k, v in expected.items()} == contract["splits"]
    assert ids[gold].tolist() == contract["gold_excluded"] and gold.sum() == 58
    assert {k: len(v) for k, v in expected.items()} == {"fit": 2637, "tune": 856, "confirmation": 852}
    assert contract["gold_report_duplicate_exclusions"] == 4
    assert report["gold_evaluations"] == report["confirmation_evaluations"] == 0
    assert report["submission_created"] is False
    db = sqlite3.connect((output / "study.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    trials = []
    for tid, number, state in db.execute("select trial_id,number,state from trials order by number"):
        attrs = {k: json.loads(v) for k, v in db.execute(
            "select key,value_json from trial_user_attributes where trial_id=?", (tid,))}
        params = {}
        for key, value, distribution in db.execute(
                "select param_name,param_value,distribution_json from trial_params where trial_id=?", (tid,)):
            dist = json.loads(distribution)
            params[key] = dist["attributes"]["choices"][int(value)] if dist["name"] == "CategoricalDistribution" else value
        values = db.execute("select value from trial_values where trial_id=?", (tid,)).fetchall()
        assert all(math.isfinite(v[0]) for v in values)
        assert state in ("COMPLETE", "PRUNED")
        checkpoint_sha = None
        if state == "COMPLETE":
            checkpoint = output / f"trial_{number}_latest.pt"
            checkpoint_sha = sha(checkpoint)
            assert checkpoint_sha == attrs["checkpoint_sha256"] and attrs["completed_epochs"] == 1
            package = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert package["epoch"] == 1 and package["params"] == params
            assert package["contract_sha256"] == sha(output / "contract.json")
            assert all(torch.isfinite(t).all().item() for t in package["model"].values() if t.is_floating_point())
            del package
        trials.append({"number": number, "state": state, "params": params,
                       "proxy_bce": -values[0][0] if values else None,
                       "seconds": attrs["seconds"], "checkpoint_sha256": checkpoint_sha})
    complete = [t for t in trials if t["state"] == "COMPLETE"]
    best = min(complete, key=lambda t: t["proxy_bce"])
    assert len(trials) == 8 and len(complete) == report["completed_trials"] == 3
    assert best["params"] == report["selected"] and best["proxy_bce"] == report["proxy_bce"]
    consumed = sum(t["seconds"] for t in trials)
    assert consumed <= contract["budget_seconds"] == 7200
    logs = list(root.glob("*.log")); assert len(logs) == 1
    log = json.loads(logs[0].read_text(encoding="utf-8"))
    text = "\n".join(r.get("data", "") for r in log)
    assert "TRANSFER_OPTUNA_COMPLETE_NO_SUBMISSION" in text
    assert "Traceback (most recent call last)" not in text
    assert not list(root.rglob("*.csv"))
    pip_warnings = [line for line in text.splitlines() if "requires " in line and "incompatible" in line]
    labels = pd.read_csv(args.labels, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    fit_y = labels.loc[contract["splits"]["fit"], TARGETS].to_numpy(float)
    tune_y = labels.loc[contract["splits"]["tune"], TARGETS].to_numpy(float)
    prior = np.clip(fit_y.mean(0), 1e-7, 1-1e-7)
    prior_bce = float(-(tune_y * np.log(prior) + (1-tune_y) * np.log1p(-prior)).mean())
    result = {"validated_checkpoints_and_sources": True, "split_counts": {k: len(v) for k, v in expected.items()},
              "active_trial_seconds": consumed, "gold_evaluations": 0, "confirmation_evaluations": 0,
              "no_submission_completion_marker": True, "fatal_traceback": False,
              "dependency_conflicts": sorted(set(pip_warnings)), "trials": trials,
              "fit_only_constant_baseline_bce": prior_bce,
              "selected_minus_constant_bce": best["proxy_bce"] - prior_bce,
              "selected_trial": best["number"], "promotion": False,
              "scope": "proxy screening only, no pathology or leaderboard improvement established",
              "artifact_hashes": {str(p.relative_to(root)): sha(p) for p in
                  (output / "contract.json", output / "screen_report.json", output / "study.sqlite3", logs[0])}}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
