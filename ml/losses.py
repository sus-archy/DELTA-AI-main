from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_softmax_loss(logits: torch.Tensor, labels: torch.Tensor, class_priors: torch.Tensor) -> torch.Tensor:
    adjusted_logits = logits + torch.log(class_priors.clamp_min(1e-12)).to(logits.device)
    return F.cross_entropy(adjusted_logits, labels)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, labels, reduction="none", weight=class_weights)
    pt = torch.exp(-ce_loss)
    loss = ((1.0 - pt) ** gamma) * ce_loss
    return loss.mean()


def class_balanced_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    return focal_loss(logits, labels, gamma=gamma, class_weights=class_weights)
