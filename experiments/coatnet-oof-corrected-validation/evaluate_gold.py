import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import train_knee as tk


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])
                          if len(np.unique(y[:, j])) > 1]))


root = next(p.parent for p in Path("/kaggle/input").rglob("train.csv")
            if (p.parent / "train_series.csv").exists())
train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
gold = train[tk.LAB].notna().all(axis=1)
gold_ids = train.loc[gold, "StudyInstanceUID"].astype(str).tolist()
truth = train.loc[gold, tk.LAB].to_numpy(np.float32)
ids = tk._open_ids()
id2row = {u: i for i, u in enumerate(ids)}
if len(gold_ids) != 58 or not set(gold_ids) <= set(ids):
    raise RuntimeError("gold identity drift")
labels = {u: truth[i] for i, u in enumerate(gold_ids)}
ds = tk.StudyWindows(tk.HERE, gold_ids, id2row, labels, 384, 24, train=False, norm="none")
dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=tk.collate)

results = {}
for name, ckpt in [("best", Path("raptor_ft_oofcorr_coatnet384_f0.pt")),
                   ("swa", Path("raptor_ft_oofcorr_coatnet384_f0_swa.pt"))]:
    if not ckpt.exists():
        continue
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    bb = tk.build_backbone(obj["arch"], pretrained=False)
    model = tk.RaptorClassifier(bb, F_dim=bb.num_features).cuda()
    model.load_state_dict(obj["model"], strict=True)
    model.eval(); pred = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred.append(torch.sigmoid(model(x.cuda()).float()).cpu().numpy())
    pred = np.concatenate(pred)
    if pred.shape != (58, 12) or not np.isfinite(pred).all():
        raise RuntimeError(f"{name} prediction integrity failure")
    np.savez(f"gold_{name}.npz", ids=np.array(gold_ids), truth=truth, pred=pred)
    results[name] = {"gold_auc": macro(truth, pred), "checkpoint_sha256": sha(ckpt)}
    del model
    torch.cuda.empty_cache()

target_audit = json.loads(Path("/kaggle/working/target_audit.json").read_text())
best_name = max(results, key=lambda k: results[k]["gold_auc"])
report = {
    "status": "COMPLETE",
    "contract": "one-fold leakage-safe OOF-corrected CoAtNet-384 screen; no submission",
    "upstream_source_sha256": "6953e8c46c714b0ee7ef6dc83dbd56232672941288953d7971f948282d92dedd",
    "target_audit": target_audit,
    "gold_fit_count": 0,
    "gold_selection_count": 0,
    "gold_evaluation_count": 58,
    "models": results,
    "best_variant": best_name,
    "best_gold_auc": results[best_name]["gold_auc"],
    "gate_auc_at_least_0_90": results[best_name]["gold_auc"] >= 0.90,
    "submission_created": False,
}
report["passed"] = bool(report["gate_auc_at_least_0_90"])
Path("/kaggle/working/coatnet_oof_corrected_report.json").write_text(json.dumps(report, indent=2) + "\n")
for p in Path("/kaggle/working").glob("*submission*.csv"):
    p.unlink()
if list(Path("/kaggle/working").glob("*submission*.csv")):
    raise RuntimeError("submission cleanup failed")
print(json.dumps(report, indent=2))

