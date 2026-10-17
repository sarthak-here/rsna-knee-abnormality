"""Deterministic quantitative features for the 288px physical knee cache."""

from __future__ import annotations

import numpy as np


def _slope(values: np.ndarray) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, values.shape[0], dtype=np.float32)
    return (values * x[:, None]).sum(0) / float((x * x).sum())


def slice_descriptors(images: np.ndarray) -> np.ndarray:
    """Return fixed intensity, texture, symmetry, and regional descriptors.

    Parameters
    ----------
    images:
        uint8 or floating array shaped ``[slice, height, width]``.
    """
    if images.ndim != 3 or images.shape[0] != 12:
        raise ValueError(f"expected [12,H,W], got {images.shape}")
    x = images.astype(np.float32) / 255.0
    flat = x.reshape(12, -1)
    quantiles = np.quantile(flat, [0.05, 0.25, 0.50, 0.75, 0.95], axis=1).T
    mean = flat.mean(1, keepdims=True)
    std = flat.std(1, keepdims=True)
    gx = np.abs(np.diff(x, axis=2)).mean((1, 2))[:, None]
    gy = np.abs(np.diff(x, axis=1)).mean((1, 2))[:, None]
    symmetry = np.abs(x - x[:, :, ::-1]).mean((1, 2))[:, None]
    bright = (x >= 0.80).mean((1, 2))[:, None]
    dark = (x <= 0.20).mean((1, 2))[:, None]

    regional = []
    ys = np.linspace(0, x.shape[1], 4).astype(int)
    xs = np.linspace(0, x.shape[2], 4).astype(int)
    for yi in range(3):
        for xi in range(3):
            cell = x[:, ys[yi]:ys[yi + 1], xs[xi]:xs[xi + 1]]
            regional.extend((cell.mean((1, 2))[:, None], cell.std((1, 2))[:, None]))
    return np.concatenate(
        [quantiles, mean, std, gx, gy, symmetry, bright, dark, *regional], axis=1
    ).astype(np.float32)


def pool_curve(descriptors: np.ndarray) -> np.ndarray:
    """Pool each 12-slice descriptor curve without learning from labels."""
    if descriptors.ndim != 2 or descriptors.shape[0] != 12:
        raise ValueError(f"expected [12,F], got {descriptors.shape}")
    return np.concatenate(
        [
            descriptors.mean(0),
            descriptors.std(0),
            descriptors.min(0),
            descriptors.max(0),
            _slope(descriptors),
        ]
    ).astype(np.float32)


def study_features(pixels: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Extract one feature vector from ``[6,12,H,W]`` physical pixels.

    Missing slots are represented by zeros plus the explicit six-bit mask. Three
    fixed paired-sequence differences add contrast-sensitive morphology signals.
    """
    if pixels.ndim != 4 or pixels.shape[:2] != (6, 12):
        raise ValueError(f"expected [6,12,H,W], got {pixels.shape}")
    masks = np.asarray(masks, dtype=np.float32)
    if masks.shape != (6,) or not np.isin(masks, (0, 1)).all():
        raise ValueError("masks must be six binary values")
    pooled = []
    for slot in range(6):
        if masks[slot]:
            pooled.append(pool_curve(slice_descriptors(pixels[slot])))
        else:
            pooled.append(None)
    width = next((len(value) for value in pooled if value is not None), None)
    if width is None:
        raise ValueError("study has no occupied slot")
    pooled = [np.zeros(width, np.float32) if value is None else value for value in pooled]
    pairs = [(0, 3), (1, 4), (2, 5)]
    differences = [
        (pooled[a] - pooled[b]) * float(masks[a] * masks[b]) for a, b in pairs
    ]
    result = np.concatenate([*pooled, *differences, masks]).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite morphology feature")
    return result
