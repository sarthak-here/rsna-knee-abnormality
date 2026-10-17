"""Fail-closed static audit for downloaded public Kaggle kernel artifacts.

This tool never executes notebook code and never submits. It binds the pulled
source, metadata, preview submission, receipts, and log into one JSON report.
An artifact audit is runtime evidence only; it does not establish a leaderboard
score unless a separate Kaggle submission identity is supplied and verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
EXPECTED_COLUMNS = ("StudyInstanceUID",) + TARGETS
PREVIEW_IDS = (
    "1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191",
    "1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776",
    "1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402",
)
COMPLETION_MARKERS = (
    "NOTEBOOK_COMPLETE",
    "CONTROLLER_DONE",
    "COMPLETE_VALIDATED",
    "validated_output_written",
)
FAILURE_MARKERS = ("Traceback (most recent call last)", "BLEND_FALLBACK", "HARD_ERROR")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_submission(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise RuntimeError("submission is empty") from error
        rows = list(reader)
    return header, rows


def receipt_summary(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid receipt JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"receipt must be an object: {path}")

    selected = {}
    for key, item in value.items():
        lowered = key.lower()
        if any(token in lowered for token in ("status", "row", "scored", "fallback", "failed", "score", "sent")):
            selected[key] = item
    return {"path": str(path), "sha256": sha256(path), "fields": selected}


def audit(
    notebook: Path,
    metadata_path: Path,
    output_dir: Path,
    expected_kernel: str | None = None,
) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("kernel metadata must be an object")
    kernel_id = metadata.get("id")
    submission_path = output_dir / "submission.csv"
    if not submission_path.is_file():
        raise RuntimeError("submission.csv is absent from downloaded outputs")

    header, rows = load_submission(submission_path)
    ids = [row[0] for row in rows if row]
    numeric = []
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(EXPECTED_COLUMNS):
            raise RuntimeError(f"row {row_number} has {len(row)} columns")
        try:
            numeric.extend(float(value) for value in row[1:])
        except ValueError as error:
            raise RuntimeError(f"row {row_number} has a non-numeric probability") from error

    logs = sorted(output_dir.glob("*.log"))
    log_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in logs)
    receipts = sorted(output_dir.rglob("*receipt*.json"))

    checks = {
        "kernel_identity": expected_kernel is None or kernel_id == expected_kernel,
        "public_kernel": metadata.get("is_private") is False,
        "internet_disabled": metadata.get("enable_internet") is False,
        "competition_attached": "rsna-knee-abnormality-detection"
        in metadata.get("competition_sources", []),
        "official_columns": tuple(header) == EXPECTED_COLUMNS,
        "preview_row_count": len(rows) == 3,
        "preview_ids_exact_order": tuple(ids) == PREVIEW_IDS,
        "preview_ids_unique": len(set(ids)) == len(ids),
        "probabilities_finite": bool(numeric) and all(math.isfinite(value) for value in numeric),
        "probabilities_in_range": bool(numeric) and all(0.0 <= value <= 1.0 for value in numeric),
        "completion_marker_present": any(marker in log_text for marker in COMPLETION_MARKERS),
        "fatal_marker_absent": not any(marker in log_text for marker in FAILURE_MARKERS),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"kernel artifact audit failed: {failed}")

    return {
        "status": "ARTIFACT_AUDIT_PASS",
        "kernel_id": kernel_id,
        "notebook_path": str(notebook),
        "notebook_sha256": sha256(notebook),
        "metadata_sha256": sha256(metadata_path),
        "submission_path": str(submission_path),
        "submission_sha256": sha256(submission_path),
        "rows": len(rows),
        "columns": header,
        "probability_min": min(numeric),
        "probability_max": max(numeric),
        "receipts": [receipt_summary(path) for path in receipts],
        "logs": [{"path": str(path), "sha256": sha256(path)} for path in logs],
        "checks": checks,
        "notebook_executed_by_auditor": False,
        "leaderboard_score_verified": False,
        "submission_created_by_auditor": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-kernel")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.notebook, args.metadata, args.output_dir, args.expected_kernel)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
