"""Predeclared soft pairwise-AUC objective for the private head screen."""

from __future__ import annotations

import torch
import torch.nn.functional as F


PAIR_THRESHOLD = 0.10
PAIR_WEIGHT = 0.85
BCE_WEIGHT = 0.15


def weighted_soft_pairwise_auc(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = PAIR_THRESHOLD,
) -> torch.Tensor:
    """Weighted logistic ranking loss over confidently ordered soft targets.

    For each target, an ordered pair contributes only when its soft-label
    difference is at least ``threshold``. Weighting by that difference prevents
    weakly separated report probabilities from dominating confident orderings.
    """

    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must have identical [batch,target] shape")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    losses = []
    for target_index in range(logits.shape[1]):
        y = targets[:, target_index].float()
        z = logits[:, target_index].float()
        target_delta = y[:, None] - y[None, :]
        keep = target_delta >= threshold
        if not bool(keep.any()):
            continue
        logit_delta = z[:, None] - z[None, :]
        weights = target_delta[keep]
        losses.append((F.softplus(-logit_delta[keep]) * weights).sum() / weights.sum())
    if not losses:
        raise RuntimeError("batch contains no target pairs above the fixed threshold")
    return torch.stack(losses).mean()


def pairwise_bce_objective(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fixed 85% ranking / 15% calibration objective."""

    ranking = weighted_soft_pairwise_auc(logits, targets)
    calibration = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    return PAIR_WEIGHT * ranking + BCE_WEIGHT * calibration
