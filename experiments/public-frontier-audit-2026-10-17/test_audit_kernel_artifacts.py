import json
from pathlib import Path

import pytest

from audit_kernel_artifacts import EXPECTED_COLUMNS, PREVIEW_IDS, audit


def write_fixture(root: Path, bad_id: bool = False) -> tuple[Path, Path, Path]:
    notebook = root / "kernel.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    metadata = root / "kernel-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "id": "owner/kernel",
                "is_private": False,
                "enable_internet": False,
                "competition_sources": ["rsna-knee-abnormality-detection"],
            }
        ),
        encoding="utf-8",
    )
    output = root / "output"
    output.mkdir()
    ids = list(PREVIEW_IDS)
    if bad_id:
        ids[-1] = "wrong"
    lines = [",".join(EXPECTED_COLUMNS)]
    lines.extend(uid + "," + ",".join(["0.5"] * 12) for uid in ids)
    (output / "submission.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "kernel.log").write_text("NOTEBOOK_COMPLETE\n", encoding="utf-8")
    (output / "receipt.json").write_text(
        json.dumps({"status": "PASS", "rows": 3, "fallback_studies": 0}),
        encoding="utf-8",
    )
    return notebook, metadata, output


def test_artifact_audit_passes_valid_preview(tmp_path: Path) -> None:
    notebook, metadata, output = write_fixture(tmp_path)
    report = audit(notebook, metadata, output, "owner/kernel")
    assert report["status"] == "ARTIFACT_AUDIT_PASS"
    assert report["rows"] == 3
    assert report["leaderboard_score_verified"] is False


def test_artifact_audit_rejects_wrong_preview_identity(tmp_path: Path) -> None:
    notebook, metadata, output = write_fixture(tmp_path, bad_id=True)
    with pytest.raises(RuntimeError, match="preview_ids_exact_order"):
        audit(notebook, metadata, output, "owner/kernel")
