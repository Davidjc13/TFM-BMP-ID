"""Tecleo: DT / FT / IKI y ventanas N=50.

NYIT solo registra pulsación (no keyup). Sin dwell medible:
DT_i = 0, FT_i = IKI_i, IKI_i = t_press[i+1] - t_press[i].
El carácter no se persiste; a lo sumo un ID hasheado en memoria.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from bmpid.features.parse_nyit import parse_keystroke_press_times
from bmpid.features.windowing import sliding_windows

MS_TO_S = 1e-3


def hash_key_id(character: str, *, salt: str = "bmpid-nyit") -> int:
    """ID no reversible para uso en memoria. No escribir el carácter."""
    digest = hashlib.sha256(f"{salt}\0{character}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def timings_from_presses(
    press_ms: np.ndarray,
    *,
    release_ms: np.ndarray | None = None,
) -> np.ndarray:
    """Devuelve [L, 3] = (DT, FT, IKI) en segundos. L = n_keys - 1."""
    t = np.asarray(press_ms, dtype=np.float64).reshape(-1)
    if t.size < 2:
        return np.zeros((0, 3), dtype=np.float32)
    iki = (t[1:] - t[:-1]) * MS_TO_S
    iki = np.maximum(iki, 0.0)
    if release_ms is None or len(release_ms) != len(t):
        dt = np.zeros_like(iki)
        ft = iki.copy()
    else:
        r = np.asarray(release_ms, dtype=np.float64).reshape(-1)
        dt = np.maximum((r[:-1] - t[:-1]) * MS_TO_S, 0.0)
        ft = (t[1:] - r[:-1]) * MS_TO_S
    return np.stack([dt, ft, iki], axis=1).astype(np.float32)


def windows_from_presses(
    press_ms: np.ndarray,
    *,
    n: int = 50,
    stride: int = 25,
    release_ms: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    seq = timings_from_presses(press_ms, release_ms=release_ms)
    return sliding_windows(seq, n=n, stride=stride)


def windows_from_csv(
    path: Path,
    *,
    n: int = 50,
    stride: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    return windows_from_presses(parse_keystroke_press_times(path), n=n, stride=stride)
