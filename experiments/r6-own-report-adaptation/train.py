"""Bounded own-report adaptation of actual R6 final encoder stage. No submission."""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

SEED = 20261020
AMP_INITIAL_SCALE = 1024.  # v1 default65536 overflow reproduced without optimization.


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prefix(model, images):
    x = (images.float() / 255. - model.mean) / model.std
    x = model.enc.stem(x.contiguous(memory_format=torch.channels_last))
    for stage in model.enc.stages[:3]:
        x = stage(x)
    return x


def tail(model, early, meta):
    x = model.enc.stages[3](early)
    x = model.enc.forward_head(model.enc.norm_pre(x))
    tokens = model.spatial_pool(x) if model.spatial else x
    mask = torch.ones((1, len(meta)), dtype=torch.bool, device=meta.device)
    return model.head(tokens, meta[None], mask, return_aux=True)


def aligned_state_prob(model, plain, reflected, perm):
    p = plain[1].float().softmax(-1)
    q = reflected[1][:, perm].float().softmax(-1)
    out = .5 * (p + q)
    out[:, 1] = p[:, 1]  # Original MCL plain-only exception.
    return out[0]


def training_objective(outputs, teacher, target, class_counts, state_valid):
    """Unchanged v1 objective, shared with the scaled actual-device preflight."""
    z, sl, bl = outputs
    known = target >= 0
    ce = F.cross_entropy(sl[0, known].float(), target[known], reduction="none")
    weights = torch.tensor(1. / class_counts[known.cpu().numpy()], dtype=torch.float32, device=z.device)
    supervision = (ce * weights).sum() * (192. / int(np.sum(class_counts > 0)))
    kl_z = F.binary_cross_entropy_with_logits(z.float(), teacher[0].sigmoid()) - F.binary_cross_entropy_with_logits(teacher[0], teacher[0].sigmoid())
    log_state = sl.float().log_softmax(-1).masked_fill(~state_valid[None], 0.)
    kl_s = F.kl_div(log_state, teacher[1].softmax(-1), reduction="none").sum(-1).mean()
    kl_b = F.binary_cross_entropy_with_logits(bl.float(), teacher[2].sigmoid()) - F.binary_cross_entropy_with_logits(teacher[2], teacher[2].sigmoid())
    return supervision + .5*kl_z + .5*kl_s + .1*kl_b


def finite_amp_step(objective, parameters, optimizer, scaler, deadline, retries):
    """Retry scaled overflow on the SAME batch/RNG; never skip a study/update.

    No model with mutable training buffers is supported by this pilot (only the
    original LayerNorm ConvNeXt final stage trains). Fail at scale1 rather than
    hiding an intrinsically nonfinite objective/gradient.
    """
    parameters = list(parameters)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if any(p.is_cuda for p in parameters) else None
    while True:
        if time.time() > deadline:
            raise TimeoutError("AMP retry reached the unchanged training deadline")
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        loss = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training objective")
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        if any(p.grad is None for p in parameters):
            raise RuntimeError("missing trainable gradient")
        if all(bool(torch.isfinite(p.grad).all()) for p in parameters):
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite unscaled gradient norm")
            scaler.step(optimizer); scaler.update()
            return float(loss.detach())
        scale = scaler.get_scale()
        retries.append({"scale": scale, "loss": float(loss.detach()), "optimizer_step_taken": False})
        if scale <= 1.:
            raise RuntimeError("nonfinite gradient at scale1; no optimizer step")
        # Explicit update clears GradScaler's per-optimizer unscale state. The
        # same forward is replayed with RNG restored, so no sample is dropped.
        scaler.update(new_scale=max(1., scale / 2.))


def state_losses(probabilities, targets):
    losses = np.full((len(targets), 12), np.nan, dtype=np.float64)
    for i, target in enumerate(targets):
        for j, state in enumerate(target):
            if state >= 0:
                losses[i, j] = -math.log(max(float(probabilities[i][j, state]), 1e-12))
    return losses


def gates(control, candidate, before, after):
    counts = np.isfinite(control).sum(0)
    eligible = counts >= 5
    gains = np.full(12, np.nan)
    for j in np.flatnonzero(eligible):
        gains[j] = np.nanmean(control[:, j] - candidate[:, j])
    delta = float(np.nanmean(gains)) if eligible.any() else 0.
    # Cluster bootstrap at the study level, retaining all known states together.
    rng = np.random.default_rng(SEED)
    boot = []
    difference = control[:, eligible] - candidate[:, eligible]
    for _ in range(2000):
        sampled = difference[rng.integers(0, len(control), len(control))]
        col = [float(np.mean(sampled[np.isfinite(sampled[:, j]), j])) for j in range(sampled.shape[1])
               if np.isfinite(sampled[:, j]).any()]
        boot.append(float(np.mean(col)) if col else 0.)
    positive = float(np.mean(np.asarray(boot) > 0))
    q025 = float(np.quantile(boot, .025))
    drift = np.abs(before - after)
    checks = {"class_coverage": int(eligible.sum()) >= 4, "macro_state_nll_gain": delta >= .01,
              "four_nonnegative_classes": int(np.sum(gains[eligible] >= 0)) >= 4,
              "bootstrap_positive": positive >= .90, "q025": q025 > -.01,
              "mean_probability_drift": float(drift.mean()) <= .05,
              "max_probability_drift": float(drift.max()) <= .30}
    return {"passed": all(checks.values()), "checks": checks, "macro_state_nll_gain": delta,
            "per_class_gain": [float(v) if np.isfinite(v) else None for v in gains],
            "known_state_counts": counts.tolist(), "bootstrap_positive": positive, "q025": q025,
            "mean_probability_drift": float(drift.mean()), "max_probability_drift": float(drift.max())}


def run(assets_path):
    a = json.loads(Path(assets_path).read_text())
    out = Path(a["out"]); out.mkdir(exist_ok=True, parents=True)
    deadline = a["started_epoch"] + 3 * 3600
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("actual training GPU preflight required")
    torch.set_num_threads(2)
    dev = "cuda:0"
    source, bundle_dir = Path(a["source"]), Path(a["bundle"])
    sys.path[:0] = [str(source), a["code"]]
    import compact_train as trainer
    import compact_infer as worker
    from postprocess import load_bundle, combine_passes, apply_postprocessors
    from prepare_splits import prepare
    split = prepare(a["train"], bundle_dir / "state_spec.json")
    if split["status"] != "READY":
        write(out / "coverage_rejection.json", split)
        return
    if sha(a["splits"]) != a["split_sha256"] or json.loads(Path(a["splits"]).read_text()) != split:
        raise ValueError("frozen local vs training metadata/splits differ")
    write(out / "contract.json", {"splits_sha256": a["split_sha256"], "epochs": 3, "seed": SEED,
          "train_groups": 192, "tune_groups": 64, "sealed_groups": 64, "windows": 32,
          "lr": 1e-5, "wd": .01, "budget_seconds": 10800, "gold_evaluation": False,
          "caveat": "report proxy and conditional image-gradient holdout, NOT whole-pipeline OOF"})
    checkpoint = bundle_dir / "E_lrprior_s0_f0.pt"
    if sha(checkpoint) != "59cc88a197bc80c7fa5faa89e69ae7bec04f69cab5259a23a0d0ceb9431e53f6":
        raise ValueError("released fold0 checkpoint changed")
    write(out / "input_identity_receipt.json", {
          "checkpoint_sha256": sha(checkpoint), "metadata_sha256": sha(a["train"]),
          "series_metadata_sha256": sha(Path(a["train"]).with_name("train_series.csv")),
          "source_sha256": {n: sha(source/n) for n in ("compact_train.py", "compact_infer.py", "px_cache.py", "knee_anchor_bank_v2.py", "side_common.py")},
          "auxiliary_sha256": {n: sha(bundle_dir/n) for n in ("state_spec.json", "side_cls.pt", "state_stacker.json", "fluid_stacker.json")},
          "anatomy_sha256": sha(a["anatomy"]), "split_sha256": a["split_sha256"],
          "own_source_sha256": {n: sha(Path(__file__).with_name(n)) for n in ("train.py", "prepare_splits.py", "report_rules.py")}})
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spec = json.loads((bundle_dir / "state_spec.json").read_text())
    reference = trainer.Net("convnext_small", None, crop=352, state_spec=spec, in_chans=3,
                            spatial="sp_q" in state, extent="ext_u" in state, syn_decomp="syn_beta" in state)
    reference.load_state_dict(state, strict=True)
    reference = reference.eval().requires_grad_(False).to(dev)
    candidate = copy.deepcopy(reference)
    candidate.enc.stages[3].requires_grad_(True)
    names = [n for n, p in candidate.named_parameters() if p.requires_grad]
    if len(names) != 31 or any(not n.startswith("enc.stages.3.") for n in names):
        raise ValueError("trainable parameter scope changed")
    post = load_bundle(source / "compact_infer.py", bundle_dir / "state_spec.json",
                       bundle_dir / "fluid_stacker.json", bundle_dir / "state_stacker.json")
    records = split["fit"] + split["tune"] + split["confirmation"]
    class_counts = np.sum(np.asarray([r["state_targets"] for r in split["fit"]]) >= 0, 0)
    cache = Path(a["cache"]); cache.mkdir(parents=True, exist_ok=True)
    ds = worker.CStudies(a["eval_root"], "test", [r["id"] for r in records], cache / "render", a["code"],
                         384, eval_stride=1, views="v4", slices=3, anchor_weights=a["anatomy"], phys_mm=3.5,
                         side_model=bundle_dir / "side_cls.pt", extent_meta=True, fluid=True)
    ds._setup()
    anchor_counts = {}
    original_record = ds.bank.record

    def checked_anchor(*args, **kwargs):
        rec = original_record(*args, **kwargs)
        if any(str(w).startswith("error:") for w in rec["why"]) or rec["method"] is None:
            raise RuntimeError("anchor error/no geometry, no silent fallback")
        anchor_counts[rec["method"]] = anchor_counts.get(rec["method"], 0) + 1
        return rec

    ds.bank.record = checked_anchor
    cache_receipts, elapsed = [], []
    parity = None
    plain_targets = {}
    for i, record in enumerate(records):
        if time.time() > deadline - 600:
            write(out / "budget_rejection.json", {"stage": "cache", "cached": i, "optimizer_steps": 0})
            return
        start = time.time()
        item = ds[i]
        if item["error"]:
            raise RuntimeError("study render failed: " + item["error"])
        n = len(item["images"])
        if n < 1:
            raise RuntimeError("empty windows")
        indices = np.rint(np.linspace(0, n - 1, 32)).astype(np.int64)
        images = item["images"][indices, :, 16:368, 16:368].to(dev)
        meta = item["meta"][indices].to(dev)
        reflected, mirror_meta = worker.mirror_windows(images[None], meta[None])
        reflected = reflected[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            early = torch.cat([prefix(reference, images[j:j+8]) for j in range(0, 32, 8)])
            mir = torch.cat([prefix(reference, reflected[j:j+8]) for j in range(0, 32, 8)])
            # ConvNeXt residuals can remain float32 under AMP. Test and use the
            # actual half-precision storage representation, not an unrounded surrogate.
            early = early.half(); mir = mir.half()
            teacher = tuple(v.float().detach() for v in tail(reference, early, meta))
            if i == 0:
                direct = reference(images[None], meta[None], torch.ones((1, 32), dtype=torch.bool, device=dev), chunk=8, return_aux=True)
                errors = [float((direct[0].float()-teacher[0]).abs().max()),
                          float((direct[1].float()[:, reference.state_valid]-teacher[1][:, reference.state_valid]).abs().max()),
                          float((direct[2].float()-teacher[2]).abs().max())]
                # Native AMP/batch choices can differ slightly; require strict finite bounded error.
                if any(not math.isfinite(v) or v > .02 for v in errors):
                    raise RuntimeError("cached prefix vs full reader parity failed: " + str(errors))
                parity = errors
            for v in (teacher[0], teacher[1][:, reference.state_valid], teacher[2], early, mir):
                if not torch.isfinite(v).all():
                    raise RuntimeError("nonfinite feature/head output")
        name = hashlib.sha256(record["id"].encode()).hexdigest()[:24] + ".npz"
        path = cache / name
        np.savez(path, early=early.cpu().numpy().astype(np.float16), mirror=mir.cpu().numpy().astype(np.float16),
                 meta=meta.cpu().numpy(), mirror_meta=mirror_meta[0].cpu().numpy(),
                 z=teacher[0].cpu().numpy(), sl=teacher[1].cpu().numpy(), bl=teacher[2].cpu().numpy())
        plain_targets[record["id"]] = path
        cache_receipts.append({"id": record["id"], "sha256": sha(path), "fluid": item["fluid"], "path": str(path),
                               "source_windows": n, "selected_window_indices": indices.tolist(),
                               "side": item["side"], "anchor": item["anchor"],
                               "feature_shape": list(early.shape), "storage_dtype": "float16"})
        torch.cuda.synchronize(); elapsed.append(time.time() - start)
        if i == 7:
            # Full forward/backward throughput measured without taking an optimizer step.
            start_step = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                probe = tail(candidate, early.detach(), meta)
            objective = training_objective(probe, teacher, torch.tensor(record["state_targets"], device=dev),
                                           class_counts, candidate.state_valid)
            (objective * AMP_INITIAL_SCALE).backward(); torch.cuda.synchronize()
            if not torch.isfinite(objective) or any(p.grad is None or not torch.isfinite(p.grad).all()
                                                   for p in candidate.parameters() if p.requires_grad):
                raise RuntimeError("scaled objective/gradient preflight failed before full cache")
            step_seconds = time.time() - start_step
            candidate.zero_grad(set_to_none=True)
            projected = time.time() + 1.5 * np.mean(elapsed) * (len(records) - 8) + 1.5 * step_seconds * (3*192 + 4*128) + 300
            preflight = {"actual_studies": 8, "study_seconds": elapsed.copy(), "training_step_seconds": step_seconds,
                         "projected_completion_epoch": projected, "deadline_epoch": deadline,
                         "optimizer_steps": 0, "prefix_full_max_abs_errors": parity,
                         "amp_initial_scale": AMP_INITIAL_SCALE, "scaled_objective_gradients_finite": True,
                         "trainable_names": names, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
            write(out / "preflight.json", preflight)
            if projected > deadline:
                write(out / "budget_rejection.json", {"stage": "preflight", **preflight})
                return
        if (i + 1) % 8 == 0:
            write(out / "cache_progress.json", {"cached": i+1, "total": len(records), "elapsed_seconds": time.time()-a["started_epoch"]})
            print("CACHE", i+1, len(records), "seconds", round(time.time()-a["started_epoch"]), flush=True)
        del images, reflected, early, mir, meta, mirror_meta, teacher, item
    write(out / "cache_receipt.json", {"entries": cache_receipts, "explicit_anchor_paths": anchor_counts, "reader_failures": 0})
    lookup = {r["id"]: r for r in cache_receipts}

    def data(record):
        with np.load(plain_targets[record["id"]], allow_pickle=False) as z:
            return {k: torch.from_numpy(z[k]).to(dev) for k in z.files}

    optimizer = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad], lr=1e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=AMP_INITIAL_SCALE)
    training_log = []
    numerical_retries = []
    for epoch in range(3):
        candidate.eval(); candidate.enc.stages[3].train()
        order = np.random.default_rng(SEED+epoch).permutation(192)
        losses = []
        for index in order:
            if time.time() > deadline - 300:
                write(out / "budget_rejection.json", {"stage": "training", "completed_epochs": epoch,
                      "completed_steps_this_epoch": len(losses), "incomplete_candidate": True})
                return
            record = split["fit"][int(index)]; d = data(record)
            target = torch.tensor(record["state_targets"], device=dev)
            def objective():
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = tail(candidate, d["early"], d["meta"])
                return training_objective(outputs, (d["z"], d["sl"], d["bl"]), target, class_counts, candidate.state_valid)
            retries = []
            try:
                loss = finite_amp_step(objective, [p for p in candidate.parameters() if p.requires_grad],
                                       optimizer, scaler, deadline-300, retries)
            except (RuntimeError, TimeoutError) as exc:
                write(out / "runtime_failure.json", {"stage": "backward", "epoch": epoch+1,
                      "completed_steps_this_epoch": len(losses), "optimizer_steps": epoch*192+len(losses),
                      "amp_scale": scaler.get_scale(), "retries": retries, "error": str(exc)})
                raise
            if retries:
                numerical_retries.append({"epoch": epoch+1, "step": len(losses)+1, "retries": retries,
                                          "successful_scale": scaler.get_scale()})
                write(out / "numerical_retry_receipt.json", numerical_retries)
            losses.append(loss)
        ckpt = out / f"epoch_{epoch+1}.pt"
        torch.save({"state_dict": candidate.cpu().state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch+1, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": (np.random.get_state()[0], np.random.get_state()[1].tolist(),
                                  *np.random.get_state()[2:]),
                    "python_rng": random.getstate()}, ckpt)
        candidate.to(dev)
        training_log.append({"epoch": epoch+1, "mean_loss": float(np.mean(losses)), "checkpoint_sha256": sha(ckpt)})
        write(out / "training_progress.json", training_log)
        print("EPOCH_COMPLETE", training_log[-1], flush=True)
    candidate.eval()
    # Assert every frozen tensor/buffer is still unchanged after all optimization.
    changed = [k for k, v in reference.state_dict().items() if not torch.equal(v, candidate.state_dict()[k])]
    if any(not k.startswith("enc.stages.3.") for k in changed):
        raise RuntimeError("frozen head/prefix/buffer changed")

    @torch.no_grad()
    def evaluate(partition):
        base_s, cand_s, base_p, cand_p = [], [], [], []
        for record in split[partition]:
            if time.time() > deadline - 30:
                raise RuntimeError("evaluation deadline exceeded; no partial advancement")
            d = data(record)
            values = []
            for model in (reference, candidate):
                with torch.autocast("cuda", dtype=torch.float16):
                    p = tail(model, d["early"], d["meta"])
                    m = tail(model, d["mirror"], d["mirror_meta"])
                p = tuple(x.float() for x in p); m = tuple(x.float() for x in m)
                sp = aligned_state_prob(model, p, m, post["perm"]).cpu().numpy()
                z, st = combine_passes(post, model, p, m)
                probs = apply_postprocessors(post, z, st, lookup[record["id"]]["fluid"])
                values.append((sp, probs))
            base_s.append(values[0][0]); base_p.append(values[0][1]); cand_s.append(values[1][0]); cand_p.append(values[1][1])
        targets = [r["state_targets"] for r in split[partition]]
        before = state_losses(base_s, targets); after = state_losses(cand_s, targets)
        result = gates(before, after, np.asarray(base_p), np.asarray(cand_p))
        np.savez_compressed(out / f"{partition}_predictions.npz", ids=np.asarray([r["id"] for r in split[partition]], dtype="U80"),
                            state_targets=np.asarray(targets, dtype=np.int16), control_states=np.asarray(base_s),
                            candidate_states=np.asarray(cand_s), control_probabilities=np.asarray(base_p), candidate_probabilities=np.asarray(cand_p))
        return result

    tuning = evaluate("tune")
    write(out / "tuning_report.json", tuning)
    confirmation = evaluate("confirmation") if tuning["passed"] else None
    final = {"status": "PASSED_PROXY_SCREEN" if confirmation and confirmation["passed"] else "REJECTED",
             "tuning": tuning, "confirmation": confirmation, "confirmation_opened": confirmation is not None,
             "training": training_log, "changed_tensors": changed, "reader_failures": 0,
             "gold_evaluated": False, "pathology_improvement_established": False,
             "submission_created": False, "larger_training_authorized": False,
             "wall_seconds": time.time()-a["started_epoch"]}
    write(out / "screen_report.json", final)
    print("OWN_REPORT_ADAPTATION_COMPLETE_NO_SUBMISSION", json.dumps(final), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--assets", required=True)
    run(p.parse_args().assets)
