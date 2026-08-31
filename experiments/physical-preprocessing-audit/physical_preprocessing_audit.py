"""Private DICOM geometry and sequence audit for the RSNA knee dataset.

This notebook does not fit a model and cannot produce a competition submission.
It measures whether a physically canonical preprocessing arm is justified before
spending GPU time on paired representation learning.
"""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


FATSAT = re.compile(r"\bfs\b|fat\s*sat|fsat|stir|spair|spir|water\s*excit|fatsup", re.I)
PD = re.compile(r"\bpd\b|\bpdw\b|proton|density|\bdp\b", re.I)
T1 = re.compile(r"\bt1\b|\bt1w\b", re.I)
T2 = re.compile(r"\bt2\b|\bt2w\b", re.I)
HEADER_TAGS = [
    "ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber",
    "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "Laterality",
    "ImageLaterality", "SeriesDescription", "SequenceName", "ScanOptions",
    "ScanningSequence", "RepetitionTime", "EchoTime", "Rows", "Columns",
]


def competition_root() -> Path:
    for candidate in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (candidate / "train.csv").is_file():
            return candidate
    raise FileNotFoundError("competition data is not attached")


def vec(value, n):
    try:
        out = np.asarray([float(x) for x in value], dtype=np.float64)
    except Exception:
        return None
    return out if len(out) >= n and np.isfinite(out[:n]).all() else None


def finite_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def header(path: Path):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=HEADER_TAGS)
    except Exception:
        return None


def classify_sequence(ds, fallback=""):
    text = " ".join(str(getattr(ds, key, "") or "") for key in (
        "SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence"
    )) if ds is not None else fallback
    tr = finite_float(getattr(ds, "RepetitionTime", None)) if ds is not None else None
    te = finite_float(getattr(ds, "EchoTime", None)) if ds is not None else None
    fs = bool(FATSAT.search(text))
    if T1.search(text) and not T2.search(text) and not PD.search(text):
        weight = "T1"
    elif T2.search(text) and not PD.search(text):
        weight = "T2"
    elif PD.search(text):
        weight = "PD"
    elif tr is not None and tr < 800:
        weight = "T1"
    elif te is not None and te > 60:
        weight = "T2"
    elif tr is not None and tr >= 800:
        weight = "PD"
    else:
        weight = "UNK"
    return weight + ("FS" if fs else "")


def audit_series(item):
    study, series, plane, description, folder = item
    files = sorted(Path(folder).glob("*.dcm"))
    row = {
        "StudyInstanceUID": study, "SeriesInstanceUID": series,
        "plane": plane, "n_slices": len(files), "readable": False,
    }
    if not files:
        return row
    # First, middle, and last detect spacing/orientation problems cheaply. Full
    # headers are read only for ordering diagnostics.
    sample_indices = sorted(set((0, len(files) // 2, len(files) - 1)))
    samples = [header(files[i]) for i in sample_indices]
    samples = [ds for ds in samples if ds is not None]
    if not samples:
        return row
    middle = samples[len(samples) // 2]
    row["readable"] = True
    row["sequence"] = classify_sequence(middle, description)
    ps = vec(getattr(middle, "PixelSpacing", None), 2)
    row["pixel_spacing_row"] = float(ps[0]) if ps is not None else None
    row["pixel_spacing_col"] = float(ps[1]) if ps is not None else None
    row["slice_thickness"] = finite_float(getattr(middle, "SliceThickness", None))
    row["spacing_between_slices"] = finite_float(getattr(middle, "SpacingBetweenSlices", None))
    tag = str(getattr(middle, "ImageLaterality", "") or getattr(middle, "Laterality", "")).upper()
    row["laterality_tag"] = tag[:1] if tag[:1] in ("L", "R") else None

    centres, normals = [], []
    for ds in samples:
        iop = vec(getattr(ds, "ImageOrientationPatient", None), 6)
        ipp = vec(getattr(ds, "ImagePositionPatient", None), 3)
        pix = vec(getattr(ds, "PixelSpacing", None), 2)
        rows = finite_float(getattr(ds, "Rows", None))
        cols = finite_float(getattr(ds, "Columns", None))
        if iop is not None:
            n = np.cross(iop[:3], iop[3:6])
            if np.linalg.norm(n) > 0:
                normals.append(n / np.linalg.norm(n))
        if iop is not None and ipp is not None and pix is not None and rows and cols:
            centre = ipp + iop[:3] * pix[1] * cols / 2 + iop[3:6] * pix[0] * rows / 2
            centres.append(centre)
    if centres:
        x = float(np.median(np.stack(centres)[:, 0]))
        row["geometry_x"] = x
        row["laterality_geometry"] = None if abs(x) < 20 else ("R" if x < 0 else "L")
    if len(normals) >= 2:
        ref = normals[0]
        row["orientation_drift_deg"] = float(max(
            np.degrees(np.arccos(np.clip(abs(float(np.dot(ref, n))), 0, 1))) for n in normals[1:]
        ))

    records = []
    for natural_index, file in enumerate(files):
        ds = header(file)
        if ds is None:
            continue
        iop = vec(getattr(ds, "ImageOrientationPatient", None), 6)
        ipp = vec(getattr(ds, "ImagePositionPatient", None), 3)
        projection = None
        if iop is not None and ipp is not None:
            normal = np.cross(iop[:3], iop[3:6])
            projection = float(np.dot(ipp, normal))
        instance = finite_float(getattr(ds, "InstanceNumber", None))
        records.append((natural_index, projection, instance))
    with_projection = [r for r in records if r[1] is not None]
    row["projection_coverage"] = len(with_projection) / max(len(files), 1)
    if len(with_projection) >= max(2, math.ceil(0.8 * len(files))):
        projections = np.asarray([r[1] for r in with_projection])
        diffs = np.diff(np.sort(projections))
        positive = diffs[diffs > 1e-4]
        if len(positive):
            row["slice_spacing_geometry"] = float(np.median(positive))
            row["slice_spacing_cv"] = float(np.std(positive) / max(np.mean(positive), 1e-6))
        projection_order = [r[0] for r in sorted(with_projection, key=lambda x: x[1])]
        instance_ready = [r for r in with_projection if r[2] is not None]
        if len(instance_ready) == len(with_projection):
            instance_order = [r[0] for r in sorted(instance_ready, key=lambda x: x[2])]
            reverse = list(reversed(instance_order))
            row["instance_projection_disagree"] = projection_order not in (instance_order, reverse)
    return row


def quantiles(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return {str(q): float(values.quantile(q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)}


def main():
    root = competition_root()
    series = pd.read_csv(root / "train_series.csv", dtype=str)
    plane_col = "Anatomical_Plane" if "Anatomical_Plane" in series.columns else "Plane"
    desc_col = "SeriesDescription" if "SeriesDescription" in series.columns else None
    series_root = root / ("train_series" if (root / "train_series").is_dir() else "train_images")
    items = []
    for row in series.itertuples(index=False):
        study = str(getattr(row, "StudyInstanceUID"))
        uid = str(getattr(row, "SeriesInstanceUID"))
        plane = str(getattr(row, plane_col, "Unknown"))
        description = str(getattr(row, desc_col, "")) if desc_col else ""
        items.append((study, uid, plane, description, series_root / study / uid))
    workers = min(16, max(2, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        audited = pd.DataFrame(pool.map(audit_series, items))
    audited.to_csv("/kaggle/working/physical_preprocessing_series_audit.csv", index=False)

    readable = audited[audited["readable"] == True].copy()
    tag_geometry = readable.dropna(subset=["laterality_tag", "laterality_geometry"])
    required = {"Sagittal": {"PDFS", "PD", "T2FS", "T2", "T1"},
                "Coronal": {"PDFS", "PD", "T2FS", "T2", "T1"},
                "Axial": {"PDFS", "PD", "T2FS", "T2"}}
    coverage = {}
    for study, group in readable.groupby("StudyInstanceUID"):
        slots = {plane: sorted(set(g["sequence"]) & allowed)
                 for plane, allowed in required.items()
                 for g in [group[group["plane"] == plane]]}
        coverage[study] = slots
    complete_pdf = sum(all(any(s.startswith(("PD", "T2")) and s.endswith("FS") for s in slots.get(p, []))
                           for p in ("Sagittal", "Coronal", "Axial")) for slots in coverage.values())
    report = {
        "contract": "header-only preflight; no fitting; no submission",
        "studies": int(series["StudyInstanceUID"].nunique()),
        "series": int(len(audited)),
        "readable_series": int(len(readable)),
        "sequence_counts": [
            {"plane": str(plane), "sequence": str(sequence), "count": int(count)}
            for (plane, sequence), count in
            readable.groupby(["plane", "sequence"]).size().sort_values(ascending=False).items()
        ],
        "pixel_spacing_row_quantiles": quantiles(readable, "pixel_spacing_row"),
        "slice_spacing_geometry_quantiles": quantiles(readable, "slice_spacing_geometry"),
        "nonuniform_spacing_series_cv_gt_005": int((pd.to_numeric(readable["slice_spacing_cv"], errors="coerce") > 0.05).sum()),
        "orientation_drift_series_gt_1deg": int((pd.to_numeric(readable["orientation_drift_deg"], errors="coerce") > 1.0).sum()),
        "instance_projection_disagreement": int((readable["instance_projection_disagree"] == True).sum()),
        "projection_coverage_lt_080": int((pd.to_numeric(readable["projection_coverage"], errors="coerce") < 0.8).sum()),
        "tag_geometry_comparable_studies": int(len(tag_geometry)),
        "tag_geometry_disagreement": int((tag_geometry["laterality_tag"] != tag_geometry["laterality_geometry"]).sum()),
        "studies_with_three_plane_fluid_sensitive_protocol": int(complete_pdf),
        "decision_gate": {
            "paired_training_required": True,
            "gold_studies_reserved_for_evaluation": 58,
            "next_step": "train identical frozen-encoder heads on incumbent and canonical representations",
        },
    }
    Path("/kaggle/working/physical_preprocessing_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

