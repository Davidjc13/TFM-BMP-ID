"""Encoder Transformer por modalidad (Pre-LN, máscaras de padding)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        t = x.size(1)
        return self.dropout(x + self.pe[:, :t, :])


class ModalityEncoder(nn.Module):
    """Proyección Linear(C → d_model) + PE + L bloques MHA/FFN."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """x [B, T, C], mask True = válido. Devuelve Z [B, T, d]."""
        h = self.pe(self.proj(x))
        pad = ~mask.bool()
        if pad.all():
            return self.norm(h)
        z = self.encoder(h, src_key_padding_mask=pad)
        return self.norm(z)


def masked_mean(z: Tensor, mask: Tensor) -> Tensor:
    m = mask.bool().unsqueeze(-1).to(dtype=z.dtype)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (z * m).sum(dim=1) / denom
