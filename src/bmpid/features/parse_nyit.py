"""Parsers numéricos del dump NYIT. El carácter de tecleo no se devuelve."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _decode_csv(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    if raw[:4] == b"\x00\x00\x00\x00":
        return []
    return raw.decode("utf-8", errors="ignore").splitlines()


def parse_keystroke_press_times(path: Path) -> np.ndarray:
    """Solo timestamps de pulsación (ms). Nunca el carácter."""
    stamps: list[float] = []
    for line in _decode_csv(path):
        line = line.strip()
        if not line:
            continue
        first = line.split(",", 1)[0].strip()
        try:
            stamps.append(float(first))
        except ValueError:
            continue
    if not stamps:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray(stamps, dtype=np.float64)


def parse_xy_series(path: Path, *, n_coords: int = 2) -> np.ndarray:
    """Filas (t, x, y[, z, ...]). n_coords=2 swipe; n_coords=3 IMU."""
    rows: list[list[float]] = []
    need = 1 + n_coords
    for line in _decode_csv(path):
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < need:
            continue
        try:
            vals = [float(parts[i]) for i in range(need)]
        except ValueError:
            continue
        rows.append(vals)
    if not rows:
        return np.zeros((0, need), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def split_gestures(series: np.ndarray, gap_ms: float) -> list[np.ndarray]:
    """Corta una serie (t, ...) en gestos por hueco temporal."""
    if series.shape[0] == 0:
        return []
    series = _dedup_timestamps(series)
    t = series[:, 0]
    cuts = np.where(np.diff(t) > float(gap_ms))[0] + 1
    idx = np.concatenate([[0], cuts, [len(series)]])
    out: list[np.ndarray] = []
    for a, b in zip(idx[:-1], idx[1:]):
        g = series[a:b]
        if len(g) >= 2:
            out.append(g)
    return out


def _dedup_timestamps(series: np.ndarray) -> np.ndarray:
    """Un punto por timestamp (el último). Evita dt=0 y derivadas explosivas."""
    if series.shape[0] < 2:
        return series
    order = np.argsort(series[:, 0], kind="mergesort")
    s = series[order]
    keep = np.ones(len(s), dtype=bool)
    keep[:-1] = np.diff(s[:, 0]) > 0
    return s[keep]
