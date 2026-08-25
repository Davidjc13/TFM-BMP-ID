"""Set2set Loss: margen entre centro de enrolamiento y consulta genuina vs impostor."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def enrollment_center(enroll: Tensor) -> Tensor:
    """enroll [B, k, d] → centro L2 [B, d]."""
    c = enroll.mean(dim=1)
    return F.normalize(c, p=2, dim=-1)


def pairwise_distance(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).norm(p=2, dim=-1)


def set2set_loss(
    enroll: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 0.2,
) -> Tensor:
    """max(0, m + d(center, q+) − d(center, q−)). Tensores ya L2-normalizados."""
    center = enrollment_center(enroll)
    d_pos = pairwise_distance(center, positive)
    d_neg = pairwise_distance(center, negative)
    return F.relu(margin + d_pos - d_neg).mean()


def similarity_score(enroll: Tensor, query: Tensor) -> Tensor:
    """Mayor = más genuino. enroll [B, k, d] o [B, d] ya promediado."""
    if enroll.dim() == 3:
        center = enrollment_center(enroll)
    else:
        center = F.normalize(enroll, p=2, dim=-1)
    return -pairwise_distance(center, query)
