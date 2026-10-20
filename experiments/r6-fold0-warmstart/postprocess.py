"""Pinned R6 TTA and deployed readout assembly; fail closed, never refit weights."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from strict_reconstruct import require, SPEC_SHA

INFER_SHA = "8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d"
FLUID_SHA = "9fbd6f652470c5ef131f997edb4e358038937dd45e6f6b65a4afed80e10cf35d"
STATE_SHA = "b05147bb463a389a66a4a0d7ff0555efd2bde6f486718f79e9d7fa3d90b24522"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def load_bundle(worker_path, spec_path, fluid_path, state_path):
    require(worker_path, INFER_SHA); require(spec_path, SPEC_SHA)
    require(fluid_path, FLUID_SHA); require(state_path, STATE_SHA)
    module_spec = importlib.util.spec_from_file_location("pinned_r6_worker", worker_path)
    worker = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(worker)
    spec = json.loads(Path(spec_path).read_text())
    layout = worker.state_layout(spec, LABELS)
    fluid = worker.fluid_stacker_load(fluid_path, 1)
    import hashlib
    state = worker.state_stacker_load(state_path, LABELS, layout, hashlib.md5(Path(spec_path).read_bytes()).hexdigest())
    if set(fluid["classes"]) & set(state["classes"]):
        raise ValueError("postprocessors overlap")
    bins = [(b, l) for b, labs in spec["binaries"].items() for l in labs]
    perm, bperm = worker.mirror_perm(LABELS, spec, bins)
    mcl_columns = []
    for column in layout["columns"]:
        if column["class"] == "MCL":
            if "categories" in column:
                mcl_columns.extend(range(column["start"], column["stop"]))
            else:
                mcl_columns.append(column["column"])
    return {"worker": worker, "spec": spec, "layout": layout, "fluid": fluid, "state": state,
            "perm": perm, "bin_perm": bperm, "bin_names": bins, "mcl_columns": mcl_columns}


def combine_passes(bundle, model, plain, mirrored):
    """One fold, batched studies: deployed logits plus deployed state readout."""
    z, sl, bl = plain; zm, slm, blm = mirrored
    if z.shape != zm.shape or z.ndim != 2 or z.shape[1] != 12:
        raise ValueError("class logit shape mismatch")
    if not torch.isfinite(z).all() or not torch.isfinite(zm).all():
        raise ValueError("nonfinite class logits")
    worker = bundle["worker"]; p = bundle["perm"]
    combined = .5 * (z + zm[:, p]); combined[:, 1] = z[:, 1]
    states = worker.state_readout(model, sl, bl)
    reflected = worker.state_readout(model, slm[:, p], blm[:, bundle["bin_perm"]])
    deployed = .5 * (states + reflected)
    deployed[:, bundle["mcl_columns"]] = states[:, bundle["mcl_columns"]]
    if not torch.isfinite(deployed).all():
        raise ValueError("nonfinite state probabilities")
    return combined, deployed


def sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


@torch.inference_mode()
def apply_postprocessors(bundle, fold_logits, fold_states, fluid_record):
    """One study; float32 fold/TTA mean before float64 stackers, as deployed."""
    if fold_logits.ndim != 2 or fold_logits.shape[1] != 12 or not torch.isfinite(fold_logits).all():
        raise ValueError("invalid fold logits")
    if fold_states.shape != (len(fold_logits), bundle["layout"]["width"]) or not torch.isfinite(fold_states).all():
        raise ValueError("invalid state readout")
    z = fold_logits.mean(0)
    probabilities = torch.sigmoid(z).double().cpu().numpy().copy()
    z64 = z.double().cpu().numpy()
    state64 = fold_states.double().mean(0).cpu().numpy()
    for name, value in bundle["worker"].fluid_stack(bundle["fluid"], LABELS, z64, fluid_record)[0].items():
        if not math.isfinite(value):
            raise ValueError("nonfinite fluid stacker result")
        probabilities[LABELS.index(name)] = sigmoid(value)
    for name, co in bundle["state"]["classes"].items():
        features = []
        for feature, column in zip(co["features"], co["columns"]):
            if feature == "z":
                features.append(float(z64[LABELS.index(name)]))
            elif feature in ("lv", "miss"):
                value = None if fluid_record is None else fluid_record.get(co["volume_column"])
                missing = value is None or not math.isfinite(float(value)) or float(value) < 0
                features.append((0. if missing else float(np.log1p(float(value)))) if feature == "lv" else float(missing))
            else:
                features.append(float(state64[column]))
        value = co["b0"] + float(np.dot(np.asarray(features, np.float64), np.asarray(co["coef"], np.float64)))
        if not math.isfinite(value):
            raise ValueError("nonfinite state stacker result")
        probabilities[LABELS.index(name)] = sigmoid(value)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("invalid final probabilities")
    return probabilities
