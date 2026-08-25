"""Fusión por atención cruzada entre tokens de modalidad (o token missing)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossModalFusion(nn.Module):
    """Un token CLS atiende a las K modalidades (keys/values)."""

    def __init__(self, d_model: int, nhead: int, dropout: float, n_modalities: int) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.missing = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.n_modalities = n_modalities

    def forward(self, tokens: Tensor, present: Tensor) -> Tensor:
        """tokens [B, M, d], present [B, M] bool. Devuelve CLS fusionado [B, d]."""
        b, m, d = tokens.shape
        miss = self.missing[:m].unsqueeze(0).expand(b, -1, -1)
        kv = torch.where(present.unsqueeze(-1), tokens, miss)
        q = self.cls.expand(b, -1, -1)
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        fused, _ = self.attn(qn, kvn, kvn, need_weights=False)
        h = q + fused
        return h.squeeze(1) + self.ff(h.squeeze(1))
