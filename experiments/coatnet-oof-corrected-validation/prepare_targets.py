import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
OOF_SHA = "60529a0ba00107abd7efd101f70454e7ce89407befbaa4107f24771ee088579e"


def one(pattern):
    hits = [p for p in Path("/kaggle/input").rglob(pattern) if p.is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"expected one {pattern}, found {hits}")
    return hits[0]


root = next(p.parent for p in Path("/kaggle/input").rglob("train.csv")
            if (p.parent / "train_series.csv").exists())
train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
ids = train.StudyInstanceUID.astype(str)
gold = train[TARGETS].notna().all(axis=1)
if len(train) != 4407 or int(gold.sum()) != 58:
    raise RuntimeError("official train identity/count drift")

sp = one("llm_labels_v4_blend.csv")
steven = pd.read_csv(sp, dtype={"StudyInstanceUID": str})
steven = train[["StudyInstanceUID"]].merge(steven, on="StudyInstanceUID", how="left", validate="one_to_one")
if not np.isfinite(steven[TARGETS].to_numpy(float)).all():
    raise RuntimeError("Steven labels incomplete")

op = one("convnext_oof_predictions.csv")
if hashlib.sha256(op.read_bytes()).hexdigest() != OOF_SHA:
    raise RuntimeError("OOF prediction SHA drift")
oof = pd.read_csv(op, dtype={"StudyInstanceUID": str})
if oof.columns.tolist() != ["StudyInstanceUID", "teacher_fold"] + TARGETS:
    raise RuntimeError("OOF schema drift")
if len(oof) != 2610 or oof.StudyInstanceUID.nunique() != 2610:
    raise RuntimeError("OOF coverage drift")
gold_ids = set(ids[gold])
if set(oof.StudyInstanceUID) & gold_ids:
    raise RuntimeError("gold leakage in teacher predictions")

targets = steven[["StudyInstanceUID"] + TARGETS].copy()
teacher = targets[["StudyInstanceUID"]].merge(oof, on="StudyInstanceUID", how="left", validate="one_to_one")
covered = teacher[TARGETS].notna().all(axis=1) & ~gold
targets.loc[covered, TARGETS] = 0.5 * targets.loc[covered, TARGETS].to_numpy(float) + 0.5 * teacher.loc[covered, TARGETS].to_numpy(float)
if int(covered.sum()) != 2610:
    raise RuntimeError("corrected coverage drift")
targets.to_parquet("/kaggle/working/labels_oof_corrected.parquet", index=False)

report_col = next((c for c in train.columns if c.lower() in {"report", "report_text", "findings"}), None)
if report_col is None:
    raise RuntimeError("radiology report column not found")
non_gold = np.flatnonzero(~gold.to_numpy())
groups = train.loc[~gold, report_col].fillna("").astype(str).map(
    lambda s: hashlib.sha256(s.strip().lower().encode()).hexdigest()).to_numpy()
splitter = GroupKFold(n_splits=5)
folds = {}
for fold, (_, va) in enumerate(splitter.split(non_gold, groups=groups)):
    for local_i in va:
        folds[str(ids.iloc[non_gold[local_i]])] = fold
if len(folds) != 4349:
    raise RuntimeError("fold coverage drift")
Path("/kaggle/working/folds.json").write_text(json.dumps({"folds": folds}, sort_keys=True))
audit = {
    "train_studies": 4407,
    "gold_evaluation_only": 58,
    "non_gold": 4349,
    "corrected": 2610,
    "unchanged": 1739,
    "teacher_folds": sorted(oof.teacher_fold.unique().astype(int).tolist()),
    "oof_sha256": OOF_SHA,
    "target_contract": "50/50 Steven v4 and true fold-held-out ConvNeXt for covered studies; Steven unchanged otherwise",
    "selection": "report-grouped fold 0 only; organizer gold excluded",
}
Path("/kaggle/working/target_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))

