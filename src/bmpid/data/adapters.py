"""Adaptadores de corpus. NYIT es el núcleo de esta fase; el resto queda documentado."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from bmpid.data.nyit import iter_manifest_rows as iter_nyit_rows


class DatasetAdapter(Protocol):
    name: str

    def iter_rows(self, root: Path, user_to_split: dict[str, str], **kwargs: Any) -> Iterator[dict[str, Any]]:
        ...


class NyitAdapter:
    name = "nyit"

    def iter_rows(self, root: Path, user_to_split: dict[str, str], **kwargs: Any) -> Iterator[dict[str, Any]]:
        return iter_nyit_rows(root, user_to_split, **kwargs)


class _UnimplementedAdapter:
    name = "unimplemented"

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def iter_rows(self, root: Path, user_to_split: dict[str, str], **kwargs: Any) -> Iterator[dict[str, Any]]:
        raise NotImplementedError(
            f"Adapter {self.name} no está en el núcleo de esta fase ({self.reason}). "
            "Corpus activo: NYIT (configs.dataset.name)."
        )


ADAPTERS: dict[str, DatasetAdapter] = {
    "nyit": NyitAdapter(),
    "aalto": _UnimplementedAdapter("aalto", "baseline TypeNet; no usado en Fase 1"),
    "humidb": _UnimplementedAdapter("humidb", "réplica SOTA / IMU opcional"),
    "behavepassdb": _UnimplementedAdapter("behavepassdb", "protocolo MobileB2C; no usado en Fase 1"),
}


def get_adapter(name: str) -> DatasetAdapter:
    key = name.lower().strip()
    if key not in ADAPTERS:
        raise KeyError(f"Dataset desconocido: {name}. Disponibles: {sorted(ADAPTERS)}")
    return ADAPTERS[key]
