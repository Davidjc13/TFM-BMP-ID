"""Sistema propuesto: encoders unimodales + fusión cross-modal → embedding L2."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bmpid.models.cross_modal import CrossModalFusion
from bmpid.models.encoders import ModalityEncoder, masked_mean

CORE = ("keystroke", "signature", "scroll")
IN_DIMS = {"keystroke": 3, "signature": 6, "scroll": 6}
MAX_LEN = {"keystroke": 64, "signature": 160, "scroll": 32}


class BehaveTransformer(nn.Module):
    def __init__(
        self,
        *,
        modalities: Sequence[str] = CORE,
        d_model: int = 128,
        d_embed: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        ff: int = 256,
        dropout: float = 0.1,
        in_dims: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        mods = tuple(m for m in modalities if m in CORE)
        if not mods:
            raise ValueError("Se requiere al menos una modalidad núcleo.")
        if "imu" in modalities:
            raise ValueError("IMU no forma parte del sistema propuesto.")
        self.modalities = mods
        dims = dict(IN_DIMS if in_dims is None else in_dims)
        self.encoders = nn.ModuleDict(
            {
                m: ModalityEncoder(
                    in_dim=int(dims[m]),
                    d_model=d_model,
                    nhead=nhead,
                    n_layers=n_layers,
                    dim_feedforward=ff,
                    dropout=dropout,
                    max_len=MAX_LEN[m],
                )
                for m in mods
            }
        )
        self.fusion = CrossModalFusion(d_model, nhead, dropout, n_modalities=len(mods))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_embed))
        self.d_embed = d_embed

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        tokens = []
        present = []
        for m in self.modalities:
            x = batch[m]
            mask = batch[f"{m}_mask"]
            flag = batch[f"{m}_present"]
            z = self.encoders[m](x, mask)
            pooled = masked_mean(z, mask)
            tokens.append(pooled)
            present.append(flag.bool())
        tok = torch.stack(tokens, dim=1)
        pres = torch.stack(present, dim=1)
        fused = self.fusion(tok, pres)
        e = self.head(fused)
        return F.normalize(e, p=2, dim=-1)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], modalities: Iterable[str] | None = None) -> BehaveTransformer:
        model_cfg = cfg.get("model") or {}
        mods = tuple(modalities) if modalities is not None else tuple((cfg.get("modalities") or {}).get("core") or CORE)
        return cls(
            modalities=mods,
            d_model=int(model_cfg.get("d_model", 128)),
            d_embed=int(model_cfg.get("d_embed", 128)),
            nhead=int(model_cfg.get("nhead", 4)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            ff=int(model_cfg.get("ff", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
