"""Experto de scroll: MLP sobre el vector de gesto 6-D."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F


class ScrollMLP(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int = 6,
        hidden: tuple[int, ...] = (64, 32),
        d_embed: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.d_embed = d_embed
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, d_embed))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if x.dim() == 3:
            if mask is None:
                feat = x.mean(dim=1)
            else:
                m = mask.bool().unsqueeze(-1).to(dtype=x.dtype)
                feat = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            feat = x
        return F.normalize(self.net(feat), p=2, dim=-1)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ScrollMLP:
        m = cfg.get("model") or {}
        e = (cfg.get("train") or {}).get("ensemble") or {}
        hidden = e.get("scroll_hidden") or [64, 32]
        return cls(
            hidden=tuple(int(h) for h in hidden),
            d_embed=int(m.get("d_embed", 128)),
            dropout=float(e.get("scroll_dropout", 0.3)),
        )
