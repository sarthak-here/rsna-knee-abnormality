"""Leakage-safe paired test of physically canonical knee-MRI preprocessing.

Public assets:
* RadImageNet ResNet-50 checkpoint, SHA-256 pinned below.
* Steven Lee Hans' v4 report-derived labels.

All 58 organizer-labelled studies are excluded from training, validation,
early stopping, and model selection. This script cannot create a submission.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50


TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
PLANES = ("Sagittal", "Coronal", "Axial")
SLOTS = (("Sagittal", True), ("Coronal", True), ("Axial", True),
         ("Sagittal", False), ("Coronal", False), ("Axial", False))
N_SLICE = 3
IMG = 224
CROP_MM = 130.0
TOKEN_DIM = 2048
SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
RAD_SHA = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
FATSAT = re.compile(r"\bfs\b|fat\s*sat|fsat|stir|spair|spir|water\s*excit|fatsup", re.I)
PD = re.compile(r"\bpd\b|\bpdw\b|proton|density|\bdp\b", re.I)
T1 = re.compile(r"\bt1\b|\bt1w\b", re.I)
T2 = re.compile(r"\bt2\b|\bt2w\b", re.I)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START = time.time()


def log(message):
    print(f"[{time.time() - START:7.1f}s] {message}", flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def competition_root():
    for path in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                 Path("/kaggle/input/rsna-knee-abnormality-detection")):
        if (path / "train.csv").is_file(): return path
    raise FileNotFoundError("competition data is not attached")


def unique_input(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name}, found {matches}")
    return matches[0]


def ffloat(value):
    try: value = float(value)
    except Exception: return None
    return value if math.isfinite(value) else None


def vec(value, n):
    try: value = np.asarray([float(x) for x in value], np.float64)
    except Exception: return None
    return value if len(value) >= n and np.isfinite(value[:n]).all() else None


def classify(ds):
    text = " ".join(str(getattr(ds, key, "") or "") for key in
                    ("SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence"))
    tr, te = ffloat(getattr(ds, "RepetitionTime", None)), ffloat(getattr(ds, "EchoTime", None))
    fs = bool(FATSAT.search(text))
    if T1.search(text) and not T2.search(text) and not PD.search(text): weight = "T1"
    elif T2.search(text) and not PD.search(text): weight = "T2"
    elif PD.search(text): weight = "PD"
    elif tr is not None and tr < 800: weight = "T1"
    elif te is not None and te > 60: weight = "T2"
    elif tr is not None and tr >= 800: weight = "PD"
    else: weight = "UNK"
    return weight, fs


def inspect_series(item):
    study, uid, plane, folder = item
    files = sorted(Path(folder).glob("*.dcm"))
    if not files: return None
    try:
        ds = pydicom.dcmread(files[len(files)//2], stop_before_pixels=True, force=True)
    except Exception: return None
    weight, fs = classify(ds)
    ps = vec(getattr(ds, "PixelSpacing", None), 2)
    iop = vec(getattr(ds, "ImageOrientationPatient", None), 6)
    ipp = vec(getattr(ds, "ImagePositionPatient", None), 3)
    rows, cols = ffloat(getattr(ds, "Rows", None)), ffloat(getattr(ds, "Columns", None))
    centre_x = None
    if ps is not None and iop is not None and ipp is not None and rows and cols:
        centre_x = float((ipp + iop[:3]*ps[1]*cols/2 + iop[3:]*ps[0]*rows/2)[0])
    tag = str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "")).upper()
    tag_lat = tag[:1] if tag[:1] in ("L", "R") else None
    geometry_lat = None if centre_x is None or abs(centre_x) < 20 else ("R" if centre_x < 0 else "L")
    # Geometry is canonical when decisive; the DICOM tag is only a fallback.
    lat = geometry_lat or tag_lat
    return {"study": study, "uid": uid, "plane": plane, "folder": str(folder),
            "files": [str(x) for x in files], "n": len(files), "weight": weight,
            "fs": fs, "ps": ps.tolist() if ps is not None else None, "lat": lat}


def choose(records, plane, fluid, canonical):
    candidates = [r for r in records if r["plane"] == plane]
    if canonical:
        if fluid:
            ranked = [(0 if r["weight"] == "PD" and r["fs"] else
                       1 if r["weight"] == "T2" and r["fs"] else
                       2 if r["fs"] else 9, -r["n"], r) for r in candidates]
        else:
            ranked = [(0 if r["weight"] == "PD" and not r["fs"] else
                       1 if r["weight"] == "T1" and not r["fs"] else
                       2 if r["weight"] == "T2" and not r["fs"] else 9,
                       -r["n"], r) for r in candidates]
        ranked = [x for x in ranked if x[0] < 9]
        return sorted(ranked, key=lambda x: (x[0], x[1]))[0][2] if ranked else None
    preferred = [r for r in candidates if bool(r["fs"]) == bool(fluid)]
    pool = preferred or candidates
    return max(pool, key=lambda r: r["n"]) if pool else None


def ordered_headers(record):
    rows = []
    for i, path in enumerate(record["files"]):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True,
                                 specific_tags=["ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber"])
            iop, ipp = vec(getattr(ds, "ImageOrientationPatient", None), 6), vec(getattr(ds, "ImagePositionPatient", None), 3)
            projection = float(np.dot(ipp, np.cross(iop[:3], iop[3:]))) if iop is not None and ipp is not None else None
            instance = ffloat(getattr(ds, "InstanceNumber", None))
        except Exception: projection, instance = None, None
        rows.append((path, projection, instance, i))
    if all(r[1] is not None for r in rows): rows.sort(key=lambda r: r[1])
    elif all(r[2] is not None for r in rows): rows.sort(key=lambda r: r[2])
    return rows


def selected_rows(record, canonical):
    rows = ordered_headers(record)
    if not rows: return []
    if not canonical:
        indices = np.linspace(.20*(len(rows)-1), .80*(len(rows)-1), N_SLICE).round().astype(int)
        return [rows[i] for i in indices]
    positions = np.asarray([r[1] if r[1] is not None else np.nan for r in rows])
    if np.isfinite(positions).all() and len(rows) > 1:
        centre = float(np.median(positions)); span = 35.0 if record["plane"] != "Axial" else 24.0
        targets = np.linspace(centre-span, centre+span, N_SLICE)
        return [rows[int(np.argmin(abs(positions-target)))] for target in targets]
    indices = np.linspace(.10*(len(rows)-1), .90*(len(rows)-1), N_SLICE).round().astype(int)
    return [rows[i] for i in indices]


def read_pixels(path):
    ds = pydicom.dcmread(path, force=True)
    image = ds.pixel_array.astype(np.float32)
    image = image * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1": image = image.max() - image
    ps = vec(getattr(ds, "PixelSpacing", None), 2)
    return image, ps


def render_slot(record, canonical):
    output = np.zeros((N_SLICE, IMG, IMG), np.uint8)
    if record is None: return output, 0
    arrays = []
    for row in selected_rows(record, canonical):
        try: arrays.append(read_pixels(row[0]))
        except Exception: arrays.append((None, None))
    valid = [a for a, _ in arrays if a is not None]
    if not valid: return output, 0
    values = np.concatenate([a.ravel()[::max(1, a.size//20000)] for a in valid])
    if canonical:
        centre = float(np.median(values)); scale = 1.4826*float(np.median(abs(values-centre)))
        if scale < 1e-6: scale = max(float(np.std(values)), 1e-6)
        low, high = centre-4*scale, centre+4*scale
    else:
        low, high = np.percentile(values, [1, 99])
    for i, (image, ps) in enumerate(arrays):
        if image is None: continue
        h, w = image.shape
        if ps is None: ps = np.asarray((.5, .5))
        if canonical:
            crop_h = min(h, max(16, int(round(CROP_MM/max(ps[0], 1e-3)))))
            crop_w = min(w, max(16, int(round(CROP_MM/max(ps[1], 1e-3)))))
        else:
            crop_h = crop_w = min(h, w, max(16, int(round(CROP_MM/max(ps[0], 1e-3)))))
        y0, x0 = (h-crop_h)//2, (w-crop_w)//2
        image = image[y0:y0+crop_h, x0:x0+crop_w]
        image = np.clip((image-low)/max(high-low, 1e-6), 0, 1)
        image = cv2.resize(image, (IMG, IMG), interpolation=cv2.INTER_AREA)
        if canonical and record.get("lat") == "R" and record["plane"] in ("Coronal", "Axial"):
            image = image[:, ::-1]
        output[i] = np.round(image*255).astype(np.uint8)
    return output, 1


class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self, x): return self.backbone(x).mean((2, 3))


def load_encoder():
    path = unique_input("ResNet50.pt")
    if sha256(path) != RAD_SHA: raise RuntimeError("RadImageNet checkpoint hash drift")
    model = Encoder(); state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True); model.eval().to(DEVICE)
    for p in model.parameters(): p.requires_grad_(False)
    return model


def encode_arm(studies, by_study, canonical):
    features = np.zeros((len(studies), len(SLOTS)*N_SLICE, TOKEN_DIM), np.float16)
    masks = np.zeros((len(studies), len(SLOTS)*N_SLICE), np.float32)
    encoder = load_encoder(); batch_study = 24
    for start in range(0, len(studies), batch_study):
        ids = studies[start:start+batch_study]
        jobs = []
        for uid in ids:
            records = by_study.get(uid, [])
            jobs.extend((choose(records, plane, fluid, canonical), canonical) for plane, fluid in SLOTS)
        with ThreadPoolExecutor(max_workers=12) as pool: rendered = list(pool.map(lambda x: render_slot(*x), jobs))
        pixels = np.stack([x[0] for x in rendered]).reshape(len(ids), len(SLOTS)*N_SLICE, IMG, IMG)
        slot_ok = np.asarray([x[1] for x in rendered], np.float32).reshape(len(ids), len(SLOTS))
        token_ok = np.repeat(slot_ok, N_SLICE, axis=1)
        flat = pixels.reshape(-1, IMG, IMG); valid = np.flatnonzero(token_ok.reshape(-1)>0)
        encoded = np.zeros((len(flat), TOKEN_DIM), np.float16)
        for b0 in range(0, len(valid), 128):
            ix = valid[b0:b0+128]
            x = torch.from_numpy(flat[ix]).to(DEVICE).float().div_(127.5).sub_(1).unsqueeze(1).expand(-1,3,-1,-1)
            with torch.autocast("cuda", enabled=DEVICE.type=="cuda"):
                y = encoder(x).float().cpu().numpy().astype(np.float16)
            encoded[ix] = y
        features[start:start+len(ids)] = encoded.reshape(len(ids), -1, TOKEN_DIM)
        masks[start:start+len(ids)] = token_ok
        if start % (batch_study*10) == 0: log(f"{'canonical' if canonical else 'incumbent'} encoded {start}/{len(studies)}")
    del encoder; gc.collect(); torch.cuda.empty_cache()
    return features, masks


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(TOKEN_DIM); self.proj=nn.Linear(TOKEN_DIM,256)
        self.att=nn.Linear(256,len(TARGETS)); self.cls=nn.Parameter(torch.empty(len(TARGETS),256)); self.bias=nn.Parameter(torch.zeros(len(TARGETS)))
        nn.init.trunc_normal_(self.cls,std=.02)
    def forward(self, x, mask):
        h=torch.nn.functional.gelu(self.proj(self.norm(x.float()))); a=self.att(h)
        a=a.masked_fill(mask[:,:,None]<.5,-1e4); a=torch.softmax(a,dim=1)
        pooled=torch.einsum("btn,btf->bnf",a,h)
        return (pooled*self.cls).sum(-1)+self.bias


def macro_auc(y, p):
    scores=[]
    for j in range(y.shape[1]):
        if len(np.unique(y[:,j]))>1: scores.append(roc_auc_score(y[:,j],p[:,j]))
    return float(np.mean(scores))


def train_head(features, masks, labels, train_idx, val_idx, seed):
    seed_all(seed); model=AttentionHead().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
    tx=torch.from_numpy(features[train_idx]); tm=torch.from_numpy(masks[train_idx]); ty=torch.from_numpy(labels[train_idx])
    loader=DataLoader(TensorDataset(tx,tm,ty),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best, best_score = None, -1
    for epoch in range(20):
        model.train()
        for x,m,y in loader:
            x,m,y=x.to(DEVICE),m.to(DEVICE),y.to(DEVICE); opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x,m),y.float()); loss.backward(); opt.step()
        pred=predict(model,features,masks,val_idx); score=macro_auc(labels[val_idx],pred)
        if score>best_score: best_score=score; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); return predict(model,features,masks,np.arange(len(features))), best_score


@torch.no_grad()
def predict(model, features, masks, indices):
    model.eval(); out=[]
    for b0 in range(0,len(indices),128):
        ix=indices[b0:b0+128]; x=torch.from_numpy(features[ix]).to(DEVICE); m=torch.from_numpy(masks[ix]).to(DEVICE)
        out.append(torch.sigmoid(model(x,m)).cpu().numpy())
    return np.concatenate(out)


def paired_bootstrap(y, inc, can, draws=5000, seed=20260901):
    rng=np.random.default_rng(seed); delta=[]
    for _ in range(draws):
        ix=rng.integers(0,len(y),len(y)); delta.append(macro_auc(y[ix],can[ix])-macro_auc(y[ix],inc[ix]))
    d=np.asarray(delta); return {"mean":float(d.mean()),"p_improve":float((d>0).mean()),"q025":float(np.quantile(d,.025)),"q975":float(np.quantile(d,.975))}


def main():
    if DEVICE.type!="cuda": raise RuntimeError("paired validation requires GPU")
    root=competition_root(); train=pd.read_csv(root/"train.csv",dtype={"StudyInstanceUID":str})
    gold=train[TARGETS].notna().all(axis=1).to_numpy();
    if int(gold.sum())!=58: raise RuntimeError(f"expected 58 gold studies, found {gold.sum()}")
    label_path=unique_input("llm_labels_v4_blend.csv"); weak=pd.read_csv(label_path,dtype={"StudyInstanceUID":str})
    aligned=train[["StudyInstanceUID"]].merge(weak,on="StudyInstanceUID",how="left",validate="one_to_one")
    labels=aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all(): raise RuntimeError("weak labels incomplete")
    series=pd.read_csv(root/"train_series.csv",dtype=str); series_root=root/("train_series" if (root/"train_series").is_dir() else "train_images")
    items=[(r.StudyInstanceUID,r.SeriesInstanceUID,r.Anatomical_Plane,series_root/r.StudyInstanceUID/r.SeriesInstanceUID) for r in series.itertuples(index=False)]
    with ThreadPoolExecutor(max_workers=16) as pool: records=[r for r in pool.map(inspect_series,items) if r]
    by_study={}
    for record in records: by_study.setdefault(record["study"],[]).append(record)
    studies=train.StudyInstanceUID.tolist(); inc_f,inc_m=encode_arm(studies,by_study,False); can_f,can_m=encode_arm(studies,by_study,True)
    nongold=np.flatnonzero(~gold); gold_idx=np.flatnonzero(gold); results=[]; inc_gold=[]; can_gold=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed); perm=rng.permutation(nongold); cut=int(.9*len(perm)); tr,va=perm[:cut],perm[cut:]
        inc_pred,inc_weak=train_head(inc_f,inc_m,labels,tr,va,seed); can_pred,can_weak=train_head(can_f,can_m,labels,tr,va,seed)
        ia=macro_auc(train.loc[gold,TARGETS].to_numpy(float),inc_pred[gold_idx]); ca=macro_auc(train.loc[gold,TARGETS].to_numpy(float),can_pred[gold_idx])
        results.append({"seed":seed,"incumbent_weak_val_auc":inc_weak,"canonical_weak_val_auc":can_weak,"incumbent_gold_auc":ia,"canonical_gold_auc":ca,"delta":ca-ia})
        inc_gold.append(inc_pred[gold_idx]); can_gold.append(can_pred[gold_idx]); log(f"seed {seed}: gold delta {ca-ia:+.6f}")
    y_gold=train.loc[gold,TARGETS].to_numpy(float); inc_mean=np.mean(inc_gold,axis=0); can_mean=np.mean(can_gold,axis=0)
    boot=paired_bootstrap(y_gold,inc_mean,can_mean); deltas=np.asarray([x["delta"] for x in results])
    gate={"delta_at_least_0_005":float(macro_auc(y_gold,can_mean)-macro_auc(y_gold,inc_mean))>=.005,
          "positive_seeds_at_least_4":int((deltas>0).sum())>=4,"bootstrap_p_at_least_0_95":boot["p_improve"]>=.95,"bootstrap_q025_positive":boot["q025"]>0}
    report={"contract":"private paired validation; gold evaluation only; no submission","encoder":"RadImageNet ResNet-50","encoder_sha256":RAD_SHA,
            "weak_labels":"stevenleehans llm_labels_v4_blend.csv","gold_fit_count":0,"gold_evaluation_count":58,"seeds":results,
            "ensemble":{"incumbent_gold_auc":macro_auc(y_gold,inc_mean),"canonical_gold_auc":macro_auc(y_gold,can_mean),"delta":macro_auc(y_gold,can_mean)-macro_auc(y_gold,inc_mean)},
            "bootstrap":boot,"gate":gate,"passed":bool(all(gate.values())),"submission_created":False}
    Path("/kaggle/working/physical_paired_validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()

