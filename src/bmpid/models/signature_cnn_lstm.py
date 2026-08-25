"""Experto de firma (proxy táctil): CNN-1D + LSTM → embedding L2."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class SignatureCNNLSTM(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int = 6,
        conv_ch: int = 32,
        hidden: int = 64,
        d_embed: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, conv_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_ch, conv_ch, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(conv_ch, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, d_embed)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        lengths = mask.bool().sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        _, (hid, _) = self.lstm(packed)
        e = self.head(torch.cat([hid[-2], hid[-1]], dim=-1))
        return F.normalize(e, p=2, dim=-1)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> SignatureCNNLSTM:
        m = cfg.get("model") or {}
        e = (cfg.get("train") or {}).get("ensemble") or {}
        return cls(
            conv_ch=int(e.get("conv_ch", 32)),
            hidden=int(e.get("hidden", 64)),
            d_embed=int(m.get("d_embed", 128)),
            dropout=float(m.get("dropout", 0.1)),
        )
