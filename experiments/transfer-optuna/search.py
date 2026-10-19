"""Optuna screening on report-group-held-out pseudo labels, never organizer gold."""
import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.nn import functional as F

from models import Reader, checked_asset, prepare_clip, sha256

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
H5_SHA = "490e5ccc07143699d05a9eb3ef9583e08795f1c2bacc6d9cf2bd0a9268158526"
STUDIES_SHA = "efc63afee30345f94a916f98ba6b30f266fccfe2976b8475fa86653b2c510513"
SEED, MAX_TRIALS, EPOCHS, BUDGET = 20261019, 8, 1, 7200


def groups_from_reports(reports):
    groups = []
    for report in reports:
        if not isinstance(report, str) or not report.strip():
            raise ValueError("missing report: cannot establish group exclusion")
        normalized = " ".join(report.casefold().split())
        groups.append(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return np.asarray(groups)


def partition(ids, groups, gold):
    if len(set(ids)) != len(ids) or len(groups) != len(ids) or len(gold) != len(ids):
        raise ValueError("identity/length drift")
    allowed = np.flatnonzero(~gold & ~np.isin(groups, groups[gold]))
    outer = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED)
    development, confirmation = next(outer.split(allowed, groups=groups[allowed]))
    development, confirmation = allowed[development], allowed[confirmation]
    inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED + 1)
    fitting, tuning = next(inner.split(development, groups=groups[development]))
    result = {"fit": development[fitting], "tune": development[tuning], "confirmation": confirmation}
    names = list(result)
    for i, name in enumerate(names):
        if set(groups[result[name]]) & set(groups[gold]):
            raise ValueError("gold report exposure")
        for other in names[i + 1:]:
            if set(groups[result[name]]) & set(groups[result[other]]):
                raise ValueError("report group overlap")
    return result


def validated_labels(train, labels):
    ids = train.StudyInstanceUID.astype(str).to_numpy()
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if len(ids) != 4407 or len(set(ids)) != 4407 or gold.sum() != 58:
        raise ValueError("official competition identity/count drift")
    aligned = train[["StudyInstanceUID"]].merge(labels, on="StudyInstanceUID", how="left", validate="one_to_one")
    # Do not read organizer targets; these are only report-derived probabilities.
    proxy = aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(proxy[~gold]).all() or (proxy[~gold] < 0).any() or (proxy[~gold] > 1).any():
        raise ValueError("invalid non-gold proxy labels")
    proxy[gold] = np.nan
    return ids, gold, proxy


class Pixels:
    def __init__(self, handle, order):
        self.handle, self.order = handle, order

    def get(self, index):
        row = self.order[index]
        mask = np.asarray(self.handle["masks"][row]).astype(bool)
        if mask.shape != (6,) or not mask.any():
            raise ValueError("missing study pixels; no fallback")
        source = np.asarray(self.handle["pixels"][row])
        clips = [prepare_clip(torch.from_numpy(source[s])) if mask[s] else torch.zeros(3, 16, 224, 224)
                 for s in range(6)]
        return torch.stack(clips)[None].cuda(), torch.from_numpy(mask)[None].cuda()


def tune_bce(model, pixels, indices, labels, deadline):
    model.eval(); total = 0.
    with torch.inference_mode():
        for index in indices:
            if time.monotonic() >= deadline:
                raise TimeoutError("hard research budget reached during validation")
            x, mask = pixels.get(index)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, mask)
            loss = F.binary_cross_entropy_with_logits(logits.float(), torch.from_numpy(labels[index:index+1]).cuda())
            if not torch.isfinite(loss):
                raise ValueError("nonfinite validation loss")
            total += loss.item()
    return total / len(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import h5py
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    assets_path = Path(args.assets)
    assets = json.loads(assets_path.read_text())
    if assets.get("terms_reviewed") is not True:
        raise ValueError("complete the model terms review before research training")
    if assets["h5"]["sha256"] != H5_SHA or assets["studies"]["sha256"] != STUDIES_SHA:
        raise ValueError("wrong physical cache release")
    paths = {key: checked_asset(assets[key]) for key in ("h5", "studies", "train", "labels")}
    # Mandatory completed hardware preflight for these exact weights.
    preflight = json.loads(checked_asset(assets["preflight"]).read_text())
    if not preflight.get("synthetic_only") or not all(r["passed"] for r in preflight["models"]):
        raise ValueError("both model execution preflights must pass")
    if {r["family"] for r in preflight["models"]} != {"swin3d_t", "videomae"}:
        raise ValueError("preflight model identity drift")
    if preflight.get("weight_hashes") != {k: assets[k]["sha256"] for k in ("swin3d_t", "videomae")}:
        raise ValueError("preflight checkpoint mismatch")
    if preflight.get("gpu") != torch.cuda.get_device_name():
        raise ValueError("rerun preflight on the actual training GPU; do not extrapolate laptop timings")
    if preflight.get("models_source_sha256") != sha256(Path(__file__).with_name("models.py")):
        raise ValueError("rerun preflight after changing model code")
    train = pd.read_csv(paths["train"], dtype={"StudyInstanceUID": str})
    labels = pd.read_csv(paths["labels"], dtype={"StudyInstanceUID": str})
    ids, gold, proxy = validated_labels(train, labels)
    groups = groups_from_reports(train.Report.tolist())
    splits = partition(ids, groups, gold)
    cache_ids = pd.read_csv(paths["studies"], dtype=str).StudyInstanceUID.tolist()
    if len(cache_ids) != 4407 or len(set(cache_ids)) != 4407 or set(cache_ids) != set(ids):
        raise ValueError("cache study identity drift")
    row = {uid: i for i, uid in enumerate(cache_ids)}
    order = np.asarray([row[uid] for uid in ids])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    contract = {"assets_sha256": sha256(assets_path), "source_sha256": sha256(__file__),
                "models_source_sha256": sha256(Path(__file__).with_name("models.py")),
                "seed": SEED, "epochs": EPOCHS, "trials": MAX_TRIALS, "budget_seconds": BUDGET,
                "objective": "negative soft report-label BCE, not pathology AUC",
                "gold_excluded": ids[gold].tolist(),
                "splits": {k: ids[v].tolist() for k, v in splits.items()},
                "gold_report_duplicate_exclusions": int((~gold & np.isin(groups, groups[gold])).sum())}
    contract_path = out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("cannot resume changed inputs/splits/source/search")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    baseline_forecast = sum(max(r["step_seconds"][1:]) for r in preflight["models"]) * (
        len(splits["fit"]) + 2 * len(splits["tune"])) * EPOCHS * 1.2
    if baseline_forecast > BUDGET:
        (out / "budget_rejection.json").write_text(json.dumps({
            "reason": "both baseline architectures cannot fit measured budget",
            "forecast_seconds": baseline_forecast, "budget_seconds": BUDGET,
            "predictive_trials": 0, "gold_evaluations": 0}, indent=2) + "\n")
        raise RuntimeError("budget infeasible: no training launched; do not increase after inspecting labels")
    study = optuna.create_study(direction="maximize", study_name="video-transfer-v1",
                                storage=f"sqlite:///{(out / 'study.sqlite3').resolve().as_posix()}",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=2),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0))
    # In-flight/failed prior runs consume their maximum lease, not zero seconds.
    consumed = sum(t.user_attrs.get("seconds", t.user_attrs.get("lease_seconds", 0)) for t in study.trials)
    remaining_trials = MAX_TRIALS - len(study.trials)
    if consumed >= BUDGET or remaining_trials <= 0:
        raise RuntimeError("fixed search budget exhausted; do not extend after results")
    deadline = time.monotonic() + BUDGET - consumed
    if not study.trials:
        for family in ("swin3d_t", "videomae"):
            study.enqueue_trial({"family": family, "lr": 3e-5, "weight_decay": 1e-3})
    with h5py.File(paths["h5"], "r") as handle:
        if handle["pixels"].shape != (4407, 6, 12, 288, 288) or handle["pixels"].dtype != np.uint8:
            raise ValueError("HDF5 pixel schema drift")
        masks = np.asarray(handle["masks"])
        if masks.shape != (4407, 6) or not np.isin(masks, (0, 1)).all() or masks.sum() != 23089:
            raise ValueError("HDF5 mask identity drift")
        pixels = Pixels(handle, order)

        def objective(trial):
            started = time.monotonic()
            trial.set_user_attr("lease_seconds", max(0, deadline - started))
            family = trial.suggest_categorical("family", ["swin3d_t", "videomae"])
            lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
            wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
            # Conservative all-slot compute forecast, before doing any training.
            timing = next(r for r in preflight["models"] if r["family"] == family)
            estimate = max(timing["step_seconds"][1:]) * (len(splits["fit"]) + 2 * len(splits["tune"]))
            if estimate * EPOCHS * 1.2 > deadline - started:
                trial.set_user_attr("seconds", time.monotonic() - started)
                raise optuna.TrialPruned("not enough measured budget for a complete one-epoch screen")
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            model = Reader(family, assets).cuda()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
            scaler = torch.amp.GradScaler("cuda")
            rng = np.random.default_rng(SEED)
            try:
                for epoch in range(EPOCHS):
                    epoch_start = time.monotonic(); model.train()
                    for seen, index in enumerate(rng.permutation(splits["fit"]), 1):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("hard study budget reached; no extra training")
                        x, mask = pixels.get(index)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.float16):
                            logits = model(x, mask)
                        loss = F.binary_cross_entropy_with_logits(logits.float(), torch.from_numpy(proxy[index:index+1]).cuda())
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite training loss")
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        scaler.step(optimizer); scaler.update()
                        if seen == (len(splits["fit"]) + 1) // 2:
                            halfway_bce = tune_bce(model, pixels, splits["tune"], proxy, deadline)
                            trial.report(-halfway_bce, 0)
                            trial.set_user_attr("seconds", time.monotonic() - started)
                            if trial.should_prune():
                                raise optuna.TrialPruned("halfway proxy-BCE median pruning")
                            model.train()
                    bce = tune_bce(model, pixels, splits["tune"], proxy, deadline)
                    checkpoint = out / f"trial_{trial.number}_latest.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(), "epoch": epoch + 1, "params": trial.params,
                                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                                "numpy_rng": rng.bit_generator.state, "python_rng": random.getstate(),
                                "contract_sha256": sha256(contract_path)}, checkpoint)
                    trial.set_user_attr("checkpoint_sha256", sha256(checkpoint))
                    trial.set_user_attr("completed_epochs", epoch + 1)
                    trial.set_user_attr("seconds", time.monotonic() - started)
                    print(json.dumps({"trial": trial.number, "family": family, "epoch": epoch + 1, "proxy_bce": bce}), flush=True)
                    trial.report(-bce, epoch + 1)
                    if trial.should_prune():
                        raise optuna.TrialPruned("median pruning on tuning groups only")
                    if epoch + 1 < EPOCHS and (time.monotonic() - epoch_start) * 1.2 > deadline - time.monotonic():
                        raise optuna.TrialPruned("measured real epoch will not fit remaining budget")
                return -bce
            finally:
                trial.set_user_attr("seconds", time.monotonic() - started)
                del model, optimizer
                gc.collect(); torch.cuda.empty_cache()

        study.optimize(objective, n_trials=remaining_trials, timeout=max(0, deadline - time.monotonic()))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    report = {"completed_trials": len(completed), "gold_evaluations": 0, "confirmation_evaluations": 0,
              "submission_created": False, "claim": "proxy screening only; no model improvement established",
              "selected": study.best_trial.params if completed else None,
              "proxy_bce": -study.best_value if completed else None}
    (out / "screen_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
