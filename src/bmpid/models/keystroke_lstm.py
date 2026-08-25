"""Experto de tecleo: BiLSTM (línea TypeNet) → embedding L2."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class KeystrokeBiLSTM(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int = 3,
        hidden: int = 64,
        n_layers: int = 2,
        d_embed: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, d_embed)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        lengths = mask.bool().sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        fwd = h[-2]
        bwd = h[-1]
        e = self.head(torch.cat([fwd, bwd], dim=-1))
        return F.normalize(e, p=2, dim=-1)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> KeystrokeBiLSTM:
        m = cfg.get("model") or {}
        e = (cfg.get("train") or {}).get("ensemble") or {}
        return cls(
            hidden=int(e.get("hidden", 64)),
            n_layers=int(e.get("lstm_layers", 2)),
            d_embed=int(m.get("d_embed", 128)),
            dropout=float(m.get("dropout", 0.1)),
        )
