import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "convnext-reader-oof-head-validation" / "convnext-reader-oof-head-validation.ipynb"
base = json.loads(BASE.read_text())
source = "".join(base["cells"][1]["source"])
prefix = source[:source.index("def new_head(seed):")]
prefix = prefix.replace(
    "from sklearn.metrics import roc_auc_score\n",
    "from sklearn.metrics import roc_auc_score\nfrom sklearn.random_projection import SparseRandomProjection\nfrom sklearn.linear_model import Ridge\n",
)
prefix = prefix.replace(
    'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n',
    'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    'ANCHOR_SHA = "6cdea905549c3df74d94c08a0547a556c9b3a890f6b1d90e432f0b79d3b0c124"\n'
    'RESIDUAL_WEIGHT = .10\nRIDGE_ALPHA = 100.0\nPROJECTION_DIM = 384\nBAG_FRACTION = .85\n',
)

tail = r'''
def ranks(p):
    out = np.empty_like(p, dtype=np.float64)
    for j in range(p.shape[1]):
        order = np.argsort(p[:, j], kind="mergesort")
        values = p[order, j]
        ranked = np.empty(len(p), np.float64)
        start = 0
        while start < len(p):
            stop = start + 1
            while stop < len(p) and values[stop] == values[start]: stop += 1
            ranked[order[start:stop]] = .5 * (start + stop - 1)
            start = stop
        out[:, j] = (ranked + .5) / len(p)
    return out

def target_aucs(y, p):
    return np.asarray([roc_auc_score(y[:, j], p[:, j]) for j in range(12)])

def logit(p):
    p = np.clip(p, .02, .98)
    return np.log(p / (1. - p))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def head_predict(model, indices):
    model.eval(); rows = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            ix = indices[start:start + 128]
            x = torch.from_numpy(features[ix]).to(DEVICE)
            m = torch.from_numpy(masks[ix]).to(DEVICE)
            p = torch.from_numpy(positions[ix]).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                rows.append(torch.sigmoid(model.head(x, m.bool(), p)).float().cpu().numpy())
    return np.concatenate(rows)

# Exact untouched public reader predictions.
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
public = KneeNet(checkpoint["args"]["backbone"].split(".")[0], pretrained=False)
public.load_state_dict(checkpoint["model"], strict=True); del public.enc
public = public.to(DEVICE).eval()
all_idx = np.arange(len(ids)); gold_idx = np.flatnonzero(gold)
public_prediction = head_predict(public, all_idx)
del public; torch.cuda.empty_cache()

# Fixed label-free sparse projection.
f = features.astype(np.float32)
slot_mean = f.mean(axis=2)
slot_std = f.std(axis=2)
representation = np.concatenate([slot_mean, slot_std], axis=2)
representation *= masks[:, :, None].astype(np.float32)
representation = representation.reshape(len(ids), -1)
projector = SparseRandomProjection(n_components=PROJECTION_DIM,
                                   dense_output=True, random_state=20261018)
z_fit = projector.fit_transform(representation[fit_idx]).astype(np.float32)
z_gold = projector.transform(representation[gold_idx]).astype(np.float32)
mean = z_fit.mean(0); std = np.maximum(z_fit.std(0), 1e-5)
z_fit = (z_fit - mean) / std; z_gold = (z_gold - mean) / std

covered_fit = np.flatnonzero(np.isin(fit_idx, covered_idx))
if len(covered_fit) != 2610: raise RuntimeError("covered fit alignment drift")
residual_target = np.clip(logit(corrected[fit_idx][covered_fit]) -
                          logit(steven[fit_idx][covered_fit]), -2., 2.)

# Immutable four-reader anchor.
anchor_path = unique_input("multireader_gold_predictions.npz")
if sha256(anchor_path) != ANCHOR_SHA: raise RuntimeError("anchor hash drift")
anchor_pkg = np.load(anchor_path, allow_pickle=False)
if anchor_pkg["ids"].astype(str).tolist() != ids[gold].tolist():
    raise RuntimeError("anchor ID drift")
truth = anchor_pkg["truth"].astype(np.float64)
if not np.array_equal(truth, train.loc[gold, TARGETS].to_numpy(np.float64)):
    raise RuntimeError("anchor truth drift")
if anchor_pkg["arm_names"].astype(str).tolist() != ["r6_flip","raptor384","raptor224","r7"]:
    raise RuntimeError("anchor arm drift")
arm = anchor_pkg["predictions"].astype(np.float64)
anchor = ranks(np.einsum("nat,a->nt", arm, np.asarray([.50,.325,.175,0.])))
anchor_auc = macro_auc(truth, anchor)
public_gold = public_prediction[gold_idx]
public_auc = macro_auc(truth, public_gold)

seed_predictions = []; seed_rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    bag = np.sort(rng.choice(covered_fit, int(BAG_FRACTION * len(covered_fit)), replace=False))
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr", tol=1e-5)
    model.fit(z_fit[bag], residual_target[bag])
    correction = np.clip(model.predict(z_gold), -1., 1.)
    corrected_gold = sigmoid(logit(public_gold) + correction)
    candidate = (1. - RESIDUAL_WEIGHT) * anchor + RESIDUAL_WEIGHT * ranks(corrected_gold)
    seed_predictions.append(corrected_gold.astype(np.float32))
    seed_rows.append({"seed": int(seed), "bag_size": int(len(bag)),
                      "reader_auc": macro_auc(truth, corrected_gold),
                      "blend_auc": macro_auc(truth, candidate),
                      "blend_minus_anchor": macro_auc(truth, candidate) - anchor_auc})
    print(seed_rows[-1], flush=True)

reader = np.mean(seed_predictions, axis=0)
candidate = (1. - RESIDUAL_WEIGHT) * anchor + RESIDUAL_WEIGHT * ranks(reader)
reader_auc = macro_auc(truth, reader); candidate_auc = macro_auc(truth, candidate)
target_delta = target_aucs(truth, candidate) - target_aucs(truth, anchor)
bootstrap = paired_bootstrap(truth, anchor, candidate, draws=10000, seed=20261018)
gate = {
    "reader_gain_at_least_0_002": reader_auc - public_auc >= .002,
    "anchor_gain_at_least_0_001": candidate_auc - anchor_auc >= .001,
    "nonnegative_targets_at_least_8": int((target_delta >= 0).sum()) >= 8,
    "positive_seeds_at_least_4": int(sum(r["blend_minus_anchor"] > 0 for r in seed_rows)) >= 4,
    "bootstrap_p_at_least_0_90": bootstrap["p_improve"] >= .90,
    "bootstrap_q025_above_minus_0_001": bootstrap["q025"] > -.001,
}
report = {
    "contract": "fixed OOF logit-residual on exact public ConvNeXt head; no submission",
    "feature_sha256": FEATURE_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
    "source_knee_sha256": KNEE_SHA, "anchor_sha256": ANCHOR_SHA,
    "fit_studies": 4349, "corrected_studies": 2610, "unchanged_studies": 1739,
    "gold_fit_count": 0, "gold_evaluation_count": 58,
    "projection": "SparseRandomProjection(seed=20261018)",
    "projection_dim": PROJECTION_DIM, "ridge_alpha": RIDGE_ALPHA,
    "bag_fraction": BAG_FRACTION,
    "residual_clip": [-1., 1.], "residual_weight": RESIDUAL_WEIGHT,
    "seed_results": seed_rows,
    "ensemble": {"public_head_auc": public_auc, "corrected_reader_auc": reader_auc,
                 "reader_gain": reader_auc-public_auc, "anchor_auc": anchor_auc,
                 "candidate_auc": candidate_auc, "candidate_gain": candidate_auc-anchor_auc,
                 "nonnegative_targets": int((target_delta >= 0).sum())},
    "bootstrap": bootstrap, "gate": gate, "passed": bool(all(gate.values())),
    "submission_created": False,
}
out = Path("/kaggle/working")
np.savez_compressed(out / "convnext_oof_logit_residual_predictions.npz",
                    ids=ids[gold].astype("U"), truth=truth.astype(np.uint8),
                    public=public_gold.astype(np.float32), anchor=anchor.astype(np.float32),
                    reader=reader.astype(np.float32), candidate=candidate.astype(np.float32),
                    seeds=np.asarray(SEEDS), seed_predictions=np.asarray(seed_predictions))
(out / "convnext_oof_logit_residual_report.json").write_text(json.dumps(report, indent=2)+"\n")
for path in out.glob("*.csv"): path.unlink()
if list(out.glob("*.csv")): raise RuntimeError("forbidden CSV artifact")
print(json.dumps(report, indent=2), flush=True)
'''

notebook = {
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {},
         "source": ["# ConvNeXt OOF logit-residual screen\n",
                    "Fixed private residual-distillation screen; organizer gold is evaluation-only; no submission.\n"]},
        {"cell_type": "code", "execution_count": None, "id": "run", "metadata": {},
         "outputs": [], "source": (prefix + tail).splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(ROOT / "convnext-oof-logit-residual.ipynb").write_text(json.dumps(notebook, indent=1))

metadata = {
    "id": "sarthaksharma14/rsna-knee-convnext-oof-logit-residual",
    "title": "RSNA Knee ConvNeXt OOF Logit Residual",
    "code_file": "convnext-oof-logit-residual.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader",
                        "stevenleehans/rsna-knee-llm-report-labels"],
    "kernel_sources": ["sarthaksharma14/rsna-knee-convnext-reader-feature-cache",
                       "sarthaksharma14/rsna-knee-convnext-oof-teacher-audit",
                       "sarthaksharma14/rsna-knee-multireader-nested-blend-screen"],
    "model_sources": [],
}
(ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")

