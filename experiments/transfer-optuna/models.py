"""Local weights only; no remote model code or silent random initialization."""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VIDEOMAE_SHA = "065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73"
VIDEOMAE_CONFIG_SHA = "6d3c7afb3d1430fa478757425cc9be2fcc50a17bfbf58921e8a9e65bb3b5c568"
SWIN_SHA = "7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checked_asset(asset, filename=None):
    path = Path(asset["path"])
    hashed = path / filename if filename else path
    if len(asset["sha256"]) != 64 or sha256(hashed) != asset["sha256"]:
        raise ValueError(f"asset identity mismatch: {hashed}")
    return path


def prepare_clip(pixels):
    """[12,H,W] uint8 -> [3,16,224,224]; explicit repeated-slice adapter."""
    if pixels.ndim != 3 or pixels.shape[0] != 12 or pixels.dtype != torch.uint8:
        raise ValueError("expected twelve uint8 physical slices")
    indices = torch.linspace(0, 11, 16, device=pixels.device).round().long()
    x = pixels[indices, None].float() / 255
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False,
                      antialias=True)[:, :, 16:240, 16:240].expand(-1, 3, -1, -1)
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return ((x - mean) / std).permute(1, 0, 2, 3).contiguous()


def migrate_video_biases(state, expected):
    """Exact old HF separate Q/V bias -> new Linear Q/K/V bias layout.

    Old keys have no K bias: zero is algebraically identical, not an initialized
    replacement parameter. No other checkpoint keys may be discarded or added.
    """
    state = dict(state)
    for key in list(state):
        if not key.endswith(".attention.attention.q_bias"):
            continue
        prefix = key.removesuffix("q_bias")
        if prefix + "query.bias" not in expected:
            continue  # Older transformers use the released layout directly.
        q, v = state.pop(key), state.pop(prefix + "v_bias")
        state[prefix + "query.bias"] = q
        state[prefix + "value.bias"] = v
        if prefix + "key.bias" in expected:
            state[prefix + "key.bias"] = torch.zeros_like(q)
    if set(state) != set(expected):
        raise ValueError(f"VideoMAE state mismatch: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}")
    return state


class Reader(nn.Module):
    def __init__(self, family, assets):
        super().__init__()
        self.family = family
        if family == "swin3d_t":
            from torchvision.models.video import swin3d_t
            path = checked_asset(assets[family])
            if assets[family]["sha256"] != SWIN_SHA:
                raise ValueError("wrong Swin release")
            self.encoder = swin3d_t(weights=None)
            self.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            self.encoder.head = nn.Identity()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.adapt = self.encoder.features[-1]
            for p in list(self.adapt.parameters()) + list(self.encoder.norm.parameters()):
                p.requires_grad_(True)
        elif family == "videomae":
            from safetensors.torch import load_file
            from transformers import VideoMAEConfig, VideoMAEForVideoClassification
            path = checked_asset(assets[family], "model.safetensors")
            if assets[family]["sha256"] != VIDEOMAE_SHA:
                raise ValueError("wrong VideoMAE release")
            if assets[family].get("revision") != "488eb9a0565f257b32866000305c8178965eb9f6":
                raise ValueError("wrong VideoMAE source revision")
            if assets[family]["config_sha256"] != VIDEOMAE_CONFIG_SHA or sha256(path / "config.json") != VIDEOMAE_CONFIG_SHA:
                raise ValueError("VideoMAE config drift")
            config = VideoMAEConfig.from_pretrained(path, local_files_only=True)
            original = VideoMAEForVideoClassification(config)
            state = migrate_video_biases(load_file(path / "model.safetensors"), original.state_dict())
            original.load_state_dict(state, strict=True)
            if original.config.num_frames != 16 or original.config.hidden_size != 768:
                raise ValueError("VideoMAE config drift")
            self.encoder = original.videomae
            self.final_norm = original.fc_norm
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.adapt = self.encoder.encoder.layer[-2:]
            for p in self.adapt.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"unknown model: {family}")
        self.slot = nn.Parameter(torch.zeros(6, 768))
        self.query = nn.Parameter(torch.randn(12, 768) * .02)
        self.weight = nn.Parameter(torch.randn(12, 768) * .02)
        self.bias = nn.Parameter(torch.zeros(12))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.adapt.train(mode)
        return self

    def encode(self, clip):
        if self.family == "swin3d_t":
            return self.encoder(clip)
        tokens = self.encoder(pixel_values=clip.permute(0, 2, 1, 3, 4)).last_hidden_state
        return self.final_norm(tokens.mean(1))

    def forward(self, clips, mask):
        # Sequential encoding bounds per-forward activation memory. Batch=1 study.
        if clips.shape[:2] != (1, 6) or mask.shape != (1, 6) or not mask.any():
            raise ValueError("expected a nonempty single-study six-slot input")
        tokens = []
        for slot in range(6):
            if mask[0, slot]:
                tokens.append(self.encode(clips[:, slot]) + self.slot[slot])
            else:
                tokens.append(self.slot[slot][None] * 0)
        token = torch.stack(tokens, 1)
        score = torch.einsum("bsd,td->bts", token, self.query) / 768 ** .5
        score = score.masked_fill(~mask[:, None], -1e4)
        context = torch.einsum("bts,bsd->btd", score.softmax(-1), token)
        return (context * self.weight).sum(-1) / 768 ** .5 + self.bias
