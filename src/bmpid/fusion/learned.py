"""Pesos de fusión aprendidos (softmax). Solo se entrena el fuser; expertos congelados."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LearnedWeights(nn.Module):
    def __init__(self, n_modalities: int = 3) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_modalities))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def weights(self) -> Tensor:
        return F.softmax(self.logits, dim=0)

    def fuse(self, scores: Tensor, present: Tensor) -> Tensor:
        w = self.weights().unsqueeze(0)
        mask = present.to(dtype=scores.dtype)
        wm = w * mask
        den = wm.sum(dim=-1).clamp(min=1e-8)
        return (scores * wm).sum(dim=-1) / den

    def forward(self, scores: Tensor, present: Tensor) -> Tensor:
        return self.scale * self.fuse(scores, present)


def fit_learned(
    scores: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 42,
) -> LearnedWeights:
    torch.manual_seed(int(seed))
    n_m = scores.shape[1]
    model = LearnedWeights(n_m)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    s = torch.from_numpy(np.asarray(scores, dtype=np.float32))
    p = torch.from_numpy(np.asarray(present, dtype=np.float32))
    t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    model.train()
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logit = model(s, p)
        loss = F.binary_cross_entropy_with_logits(logit, t)
        loss.backward()
        opt.step()
    model.eval()
    return model
