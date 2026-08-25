"""Conteo de eventos en CSV NYIT sin persistir caracteres ni coordenadas."""

from __future__ import annotations

from pathlib import Path


def _first_field_float(line: str) -> float | None:
    if not line:
        return None
    first = line.split(",", 1)[0].strip()
    if not first:
        return None
    try:
        return float(first)
    except ValueError:
        return None


def iter_event_timestamps(path: Path) -> list[float]:
    """Timestamps de filas numéricas. No devuelve el resto de la línea (texto / x,y)."""
    stamps: list[float] = []
    try:
        raw = path.read_bytes()
    except OSError:
        return stamps
    if raw[:4] == b"\x00\x00\x00\x00":
        return stamps
    text = raw.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        ts = _first_field_float(line.strip())
        if ts is not None:
            stamps.append(ts)
    return stamps


def count_timestamped_rows(path: Path) -> int:
    return len(iter_event_timestamps(path))


def count_swipe_gestures(path: Path, gap_ms: float = 300.0) -> tuple[int, int]:
    """Número de puntos y de gestos (cortes por hueco temporal)."""
    stamps = iter_event_timestamps(path)
    n_points = len(stamps)
    if n_points == 0:
        return 0, 0
    n_gestures = 1
    prev = stamps[0]
    for ts in stamps[1:]:
        if ts - prev > gap_ms:
            n_gestures += 1
        prev = ts
    return n_points, n_gestures
