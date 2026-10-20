"""Bounded official preview DICOM fixture, CPU parity only; no training labels."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    competition = "rsna-knee-abnormality-detection"
    # First page contains the first preview series followed by later series.
    # Refuse if the selected series reaches the page boundary: never label a
    # truncated series complete or follow into large training archives.
    response = api.competition_list_files(competition, page_size=200)
    names = [f.name for f in response.files]
    dcms = [f for f in response.files if f.name.startswith("test_series/") and f.name.endswith(".dcm")]
    if not dcms:
        raise ValueError("no official preview DICOMs in bounded listing")
    prefix = dcms[0].name.rsplit("/", 1)[0] + "/"
    selected = [f for f in dcms if f.name.startswith(prefix)]
    if response.next_page_token and names[-1].startswith(prefix):
        raise ValueError("series extends beyond listing; no partial download")
    # The service's actual byte field is total_bytes (CLI display is not used).
    sizes = [int(f.total_bytes) for f in selected]
    if not 3 <= len(selected) <= 80 or sum(sizes) > 96 * 1024 ** 2:
        raise ValueError("preview fixture exceeds 80 files / 96 MiB cap")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("test.csv", "test_series.csv"):
        api.competition_download_file(competition, name, path=str(root), quiet=True)
    records = []
    for file, expected_bytes in zip(selected, sizes):
        target = root / file.name
        if root not in target.resolve().parents:
            raise ValueError("unsafe fixture path")
        target.parent.mkdir(parents=True, exist_ok=True)
        api.competition_download_file(competition, file.name, path=str(target.parent), quiet=True)
        if not target.is_file() or target.stat().st_size != expected_bytes:
            raise ValueError("downloaded DICOM size mismatch")
        records.append({"path": file.name, "bytes": expected_bytes,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    report = {"source": competition, "split": "public test preview", "purpose": "CPU preprocessing parity only",
              "series_prefix": prefix, "files": records, "bytes": sum(sizes),
              "source_listing_page_size": 200, "source_boundary_check": "selected prefix ends before first page boundary",
              "scope": "one preview series, not an entire study or independently labelled evaluation",
              "gpu_used": False, "training_labels_downloaded": False}
    (root / "preview_fixture_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    print(f"Downloaded {len(records)} DICOM files")


if __name__ == "__main__":
    main()
