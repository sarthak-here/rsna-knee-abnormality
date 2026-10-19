"""Fetch two official releases for execution checks, not medical training data."""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from models import VIDEOMAE_SHA, sha256

REVISION = "488eb9a0565f257b32866000305c8178965eb9f6"
SWIN_URL = "https://download.pytorch.org/models/swin3d_t-7615ae03.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    swin = out / "swin3d_t-7615ae03.pth"
    if not swin.exists():
        torch.hub.download_url_to_file(SWIN_URL, str(swin), hash_prefix="7615ae03")
    if not sha256(swin).startswith("7615ae03"):
        raise ValueError("official Swin filename hash mismatch")
    video = out / "videomae"
    for filename in ("config.json", "preprocessor_config.json", "model.safetensors", "README.md"):
        hf_hub_download("MCG-NJU/videomae-base-finetuned-kinetics", filename,
                        revision=REVISION, local_dir=video)
    if sha256(video / "model.safetensors") != VIDEOMAE_SHA:
        raise ValueError("VideoMAE weight identity mismatch")
    config_sha = sha256(video / "config.json")
    assets = {"terms_reviewed": False,
              "purpose": "execution preflight only; not cleared for competition submission",
              "swin3d_t": {"path": str(swin), "sha256": sha256(swin), "source": SWIN_URL},
              "videomae": {"path": str(video), "sha256": VIDEOMAE_SHA,
                           "config_sha256": config_sha, "revision": REVISION,
                           "license": "CC-BY-NC-4.0"}}
    (out / "assets.json").write_text(json.dumps(assets, indent=2) + "\n")
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
