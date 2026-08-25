"""Muestreo Set2set: k anclas + positivo del mismo usuario + negativo de otro."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from bmpid.train.dataset import NyitFeatureStore


def _stack_views(views: list[dict[str, Any]], modalities: Sequence[str]) -> dict[str, Tensor]:
    batch: dict[str, Tensor] = {}
    for m in modalities:
        batch[m] = torch.from_numpy(np.stack([v[m] for v in views], axis=0))
        batch[f"{m}_mask"] = torch.from_numpy(np.stack([v[f"{m}_mask"] for v in views], axis=0))
        batch[f"{m}_present"] = torch.tensor([v[f"{m}_present"] for v in views], dtype=torch.bool)
    return batch


class Set2setSampler:
    def __init__(
        self,
        store: NyitFeatureStore,
        split: str,
        *,
        k_enroll: int,
        rng: np.random.Generator,
        require_all: bool = False,
    ) -> None:
        self.store = store
        self.split = split
        self.k_enroll = int(k_enroll)
        self.rng = rng
        self.users = store.users(split, require_all=require_all)
        if len(self.users) < 2:
            raise ValueError(f"Set2set necesita ≥2 usuarios en {split}.")

    def _one_view(self, user_id: str) -> dict[str, Any]:
        view: dict[str, Any] = {}
        for m in self.store.modalities:
            n = self.store.n_items(m, self.split, user_id)
            if n == 0:
                x, mask = self.store.empty_view(m)
                view[m] = x
                view[f"{m}_mask"] = mask
                view[f"{m}_present"] = False
                continue
            idx = int(self.rng.integers(0, n))
            x, mask, present = self.store.fetch(m, self.split, user_id, idx)
            view[m] = x
            view[f"{m}_mask"] = mask
            view[f"{m}_present"] = present
        return view

    def sample_pack(self, batch_size: int) -> dict[str, Tensor]:
        views: list[dict[str, Any]] = []
        users_arr = np.asarray(self.users)
        chosen = self.rng.choice(users_arr, size=batch_size, replace=len(self.users) < batch_size)
        for uid in chosen:
            others = np.asarray([u for u in self.users if u != uid])
            neg_uid = str(self.rng.choice(others))
            for _ in range(self.k_enroll):
                views.append(self._one_view(str(uid)))
            views.append(self._one_view(str(uid)))
            views.append(self._one_view(neg_uid))
        packed = _stack_views(views, self.store.modalities)
        packed["batch_size"] = torch.tensor(batch_size)
        packed["k_enroll"] = torch.tensor(self.k_enroll)
        return packed


def unpack_set2set(
    embeddings: Tensor,
    batch_size: int,
    k_enroll: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """embeddings [B*(k+2), d] → enroll [B,k,d], pos [B,d], neg [B,d]."""
    d = embeddings.size(-1)
    grouped = embeddings.view(batch_size, k_enroll + 2, d)
    enroll = grouped[:, :k_enroll, :]
    pos = grouped[:, k_enroll, :]
    neg = grouped[:, k_enroll + 1, :]
    return enroll, pos, neg


def views_to_batch(views: Sequence[Mapping[str, Any]], modalities: Sequence[str]) -> dict[str, Tensor]:
    return _stack_views(list(views), modalities)


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for key, val in batch.items():
        out[key] = val.to(device) if isinstance(val, Tensor) else val
    return out
