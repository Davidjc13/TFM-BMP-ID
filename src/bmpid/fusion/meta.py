"""Meta-clasificador sobre scores + máscaras de modalidad."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MetaFuser(nn.Module):
    def __init__(self, n_modalities: int = 3, hidden: int = 16) -> None:
        super().__init__()
        in_dim = n_modalities * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def features(self, scores: Tensor, present: Tensor) -> Tensor:
        return torch.cat([scores, present.to(dtype=scores.dtype)], dim=-1)

    def forward(self, scores: Tensor, present: Tensor) -> Tensor:
        return self.net(self.features(scores, present)).squeeze(-1)


def fit_meta(
    scores: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 400,
    lr: float = 0.02,
    seed: int = 42,
    hidden: int = 16,
) -> MetaFuser:
    torch.manual_seed(int(seed))
    n_m = scores.shape[1]
    model = MetaFuser(n_m, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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
