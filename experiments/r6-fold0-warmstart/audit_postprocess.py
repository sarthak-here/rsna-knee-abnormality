"""CPU synthetic parity against exact worker function bodies, no study scoring."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from postprocess import load_bundle, combine_passes, apply_postprocessors, LABELS, INFER_SHA, STATE_SHA, FLUID_SHA
from strict_reconstruct import TRAINER_SHA, require


def extract(source, names, env):
    tree = ast.parse(source)
    definitions = []
    for name in names:
        hits = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(hits) != 1:
            raise ValueError(f"cannot identify exact source function {name}")
        definitions.append(hits[0])
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "pinned-worker-function-bodies", "exec"), env)


class Sink:
    def writerow(self, row):
        pass

    def flush(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = Path(args.bundle_dir)
    bundle = load_bundle(args.worker, folder / "state_spec.json", folder / "fluid_stacker.json", folder / "state_stacker.json")
    require(args.trainer, TRAINER_SHA)
    spec = importlib.util.spec_from_file_location("parity_original_trainer", args.trainer)
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    worker = bundle["worker"]
    assert trainer.state_layout(bundle["spec"]) == bundle["layout"]
    assert trainer.mirror_perm(bundle["spec"], bundle["bin_names"]) == (bundle["perm"], bundle["bin_perm"])
    image = torch.arange(1 * 6 * 3 * 5 * 5, dtype=torch.float32).reshape(1, 6, 3, 5, 5)
    meta = torch.zeros(1, 6, 9); meta[0, :, 7] = torch.tensor([0, 1, 1, 2, 0, 2])
    meta[0, :, 1] = torch.tensor([0, 2, 3, 4, 5, 1]); meta[0, :, 2] = torch.linspace(-1, 1, 6)
    image[0, -1] = 0; meta[0, -1] = 0
    reference = trainer.mirror_batch(image, meta); copied = worker.mirror_windows(image, meta)
    assert all(torch.equal(a, b) for a, b in zip(reference, copied))
    twice = worker.mirror_windows(*copied)
    assert torch.equal(twice[0], image) and torch.equal(twice[1], meta)
    assert torch.equal(copied[0][0, 0], image[0, 0].flip(0))
    assert torch.equal(copied[0][0, 1], image[0, 1].flip(-1))
    k = [len(bundle["spec"]["categories"][l]) for l in LABELS]
    kmax = max(k); nbin = len(bundle["bin_names"])
    generator = torch.Generator().manual_seed(20261020)
    max_error = 0.; tta_cases = 0; output_cases = 0
    for fold_count in (1, 3):
        for draw in range(8):
            models = []
            for _ in range(fold_count):
                def sample():
                    z = torch.randn(1, 12, generator=generator) * (20 if draw == 7 else 2)
                    sl = torch.randn(1, 12, kmax, generator=generator)
                    for j, width in enumerate(k):
                        sl[:, j, width:] = -torch.inf
                    return z, sl, torch.randn(1, nbin, generator=generator)
                models.append(SimpleNamespace(state_k=k, bin_names=bundle["bin_names"], plain=sample(), mirrored=sample()))
            def one_pass(model, pixels, geometry, mask):
                return model.plain if torch.equal(pixels, image) else model.mirrored
            extra = {}
            env = {"torch": torch, "np": np, "math": math, "FLIP": {"perm": bundle["perm"], "mcl": 1, "bin_perm": bundle["bin_perm"]},
                   "MCL_STATE_COLS": bundle["mcl_columns"], "EXTRA": extra,
                   "one_pass": one_pass, "mirror_windows": worker.mirror_windows, "state_readout": worker.state_readout}
            extract(Path(args.worker).read_text(), ["score_v4c", "apply_stacker", "apply_state_stacker"], env)
            reference_z = torch.stack(env["score_v4c"](models, image, meta, torch.ones(1, 6, dtype=torch.bool)))
            got = [combine_passes(bundle, m, m.plain, m.mirrored) for m in models]
            actual_z = torch.cat([r[0] for r in got]); actual_s = torch.cat([r[1] for r in got])
            reference_s = torch.stack(extra["group1"]["state"])
            assert torch.equal(actual_z, reference_z) and torch.equal(actual_s, reference_s)
            for model, (z, st) in zip(models, got):
                assert torch.equal(z[:, 1], model.plain[0][:, 1])
                assert torch.equal(st[:, bundle["mcl_columns"]], worker.state_readout(model, *model.plain[1:])[:, bundle["mcl_columns"]])
            tta_cases += 1
            for volume in (None, 0., 2.5, 1000., -1., float("nan")):
                fluid = None if volume is None else {"status": "synthetic", "eff_sag_p99_open": volume, "bak_sag_p99_open": volume}
                env.update({"LABELS": LABELS, "STK": bundle["fluid"], "SST": bundle["state"],
                            "models": models, "models2": [], "fluid_stack": worker.fluid_stack,
                            "SS": {"studies": 0, "stack_fallback": {}},
                            "FL": {"studies": 0, "status": {}, "fluid_s": 0., "stack_fallback": {}, "first": [],
                                   "with_volume": {l: 0 for l in bundle["fluid"]["classes"]},
                                   "missing": {l: 0 for l in bundle["fluid"]["classes"]}},
                            "swr": Sink(), "sfh": Sink(), "fwr": Sink(), "ffh": Sink()})
                probabilities = torch.sigmoid(reference_z.mean(0)).double().numpy().copy()
                env["apply_stacker"]("synthetic", {"fluid": fluid}, reference_z, probabilities)
                env["apply_state_stacker"]("synthetic", {"fluid": fluid}, reference_z, probabilities)
                assert not env["SS"]["stack_fallback"] and not env["FL"]["stack_fallback"]
                actual = apply_postprocessors(bundle, actual_z, actual_s, fluid)
                error = float(np.abs(actual - probabilities).max()); max_error = max(max_error, error)
                assert error <= 1e-12
                output_cases += 1
    bad = actual_z.clone(); bad[0, 0] = torch.nan
    try:
        apply_postprocessors(bundle, bad, actual_s, None)
    except ValueError:
        nonfinite_rejected = True
    else:
        raise AssertionError("nonfinite logits accepted")
    result = {"scope": "synthetic CPU postprocessing, not raw-DICOM or study inference parity",
              "fixture_seed": 20261020, "torch_version": torch.__version__, "numpy_version": np.__version__,
              "implementation_sha256": hashlib.sha256(Path(__file__).with_name("postprocess.py").read_bytes()).hexdigest(),
              "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "worker_sha256": INFER_SHA, "trainer_sha256": TRAINER_SHA,
              "fluid_stacker_sha256": FLUID_SHA, "state_stacker_sha256": STATE_SHA,
              "trainer_worker_mirror_and_state_layout_exact": True, "mirror_involution": True,
              "tta_fixture_cases": tta_cases, "tta_logits_states_bitwise_equal": True,
              "mcl_plain_only_logits_states_binaries_verified": True,
              "postprocessing_fixture_cases": output_cases, "max_abs_probability_difference": max_error,
              "source_silent_fallbacks_during_fixtures": 0, "nonfinite_inputs_rejected": nonfinite_rejected,
              "fluid_adopted_classes": list(bundle["fluid"]["classes"]),
              "state_adopted_classes": list(bundle["state"]["classes"]),
              "gpu_used": False, "optimizer_steps": 0, "study_predictions": 0,
              "raw_dicom_parity_tested": False, "fluid_measurement_from_pixels_tested": False,
              "promotion": False}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
