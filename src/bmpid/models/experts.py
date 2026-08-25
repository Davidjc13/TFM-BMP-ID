"""Adaptador: experto unimodal con el mismo batch dict que el Transformer."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from bmpid.models.keystroke_lstm import KeystrokeBiLSTM
from bmpid.models.scroll_mlp import ScrollMLP
from bmpid.models.signature_cnn_lstm import SignatureCNNLSTM

CORE = ("keystroke", "signature", "scroll")


class ExpertWrapper(nn.Module):
    def __init__(self, expert: nn.Module, modality: str) -> None:
        super().__init__()
        self.expert = expert
        self.modality = modality
        self.d_embed = int(getattr(expert, "d_embed", 128))

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        m = self.modality
        return self.expert(batch[m], batch[f"{m}_mask"])


def build_expert(cfg: dict[str, Any], modality: str) -> ExpertWrapper:
    if modality == "keystroke":
        inner = KeystrokeBiLSTM.from_config(cfg)
    elif modality == "signature":
        inner = SignatureCNNLSTM.from_config(cfg)
    elif modality == "scroll":
        inner = ScrollMLP.from_config(cfg)
    else:
        raise ValueError(f"Modalidad no núcleo: {modality}")
    return ExpertWrapper(inner, modality)
