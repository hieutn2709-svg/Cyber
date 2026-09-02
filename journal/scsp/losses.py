"""Losses for severe relation imbalance in the Gate A plain model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if pos_weight <= 0:
        raise ValueError("pos_weight must be > 0")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    targets = targets.to(dtype=logits.dtype)
    positive_weight = logits.new_tensor(pos_weight)
    base = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
        reduction="none",
    )
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    focal = (1.0 - p_t).pow(gamma)
    return (focal * base).mean()


def positive_relation_type_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("relation type logits must have shape [pairs, relation_types]")
    if labels.ndim != 1 or positive_mask.ndim != 1:
        raise ValueError("labels and positive_mask must have shape [pairs]")
    if logits.shape[0] != labels.shape[0] or labels.shape[0] != positive_mask.shape[0]:
        raise ValueError("pair dimension must agree across logits, labels, and mask")
    mask = positive_mask.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], labels.long()[mask])
