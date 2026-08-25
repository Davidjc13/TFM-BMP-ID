"""Scroll / swipe: un vector de gesto (ensemble) o tokens por gesto (Transformer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bmpid.features.parse_nyit import parse_xy_series, split_gestures

MS_TO_S = 1e-3
SCROLL_DIM = 6


def scroll_gesture_to_vector(gesture: np.ndarray) -> np.ndarray:
    """Features [6]: v_mean, v_max, a_mean, longitud euclidiana, duración, desv. angular.

    Eje vertical = (0, 1) en coordenadas de pantalla. Un trazo vertical puro
    tiene desviación ~0. `gesture` es [P, >=3] (t_ms, x, y).
    """
    g = np.asarray(gesture, dtype=np.float64)
    out = np.zeros((SCROLL_DIM,), dtype=np.float32)
    if g.shape[0] < 2:
        return out
    t = (g[:, 0] - g[0, 0]) * MS_TO_S
    duration = float(max(t[-1], 0.0))
    x = g[:, 1]
    y = g[:, 2]
    step = np.hypot(np.diff(x), np.diff(y))
    length = float(np.sum(step))
    disp = np.array([x[-1] - x[0], y[-1] - y[0]], dtype=np.float64)
    nrm = float(np.linalg.norm(disp))
    if nrm < 1e-8:
        ang = 0.0
    else:
        vertical = np.array([0.0, 1.0], dtype=np.float64)
        cos = float(np.clip(abs(np.dot(disp, vertical)) / nrm, 0.0, 1.0))
        ang = float(np.arccos(cos))
    if duration < 1e-3:
        v_mean = v_max = a_mean = 0.0
    else:
        n = max(2, int(round(duration * 100.0)) + 1)
        tg = np.linspace(0.0, duration, n)
        xi = np.interp(tg, t, x)
        yi = np.interp(tg, t, y)
        dt = float(tg[1] - tg[0]) if n > 1 else 0.01
        speed = np.hypot(np.gradient(xi, dt), np.gradient(yi, dt))
        acc = np.gradient(speed, dt)
        v_mean = float(speed.mean())
        v_max = float(speed.max())
        a_mean = float(acc.mean())
    out[0] = v_mean
    out[1] = v_max
    out[2] = a_mean
    out[3] = length
    out[4] = duration
    out[5] = ang
    return out


def vectors_from_csv(path: Path, *, gap_ms: float = 300.0) -> np.ndarray:
    series = parse_xy_series(path, n_coords=2)
    gestures = split_gestures(series, gap_ms)
    if not gestures:
        return np.zeros((0, SCROLL_DIM), dtype=np.float32)
    return np.stack([scroll_gesture_to_vector(g) for g in gestures], axis=0)


def tokens_from_vectors(
    vectors: np.ndarray,
    *,
    max_gestures: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Un trial → [G, F] + máscara [G]. Padding si hay menos gestos."""
    v = np.asarray(vectors, dtype=np.float32)
    g = int(max_gestures)
    f = SCROLL_DIM
    out = np.zeros((g, f), dtype=np.float32)
    mask = np.zeros((g,), dtype=np.uint8)
    n = min(g, v.shape[0])
    if n:
        out[:n] = v[:n]
        mask[:n] = 1
    return out, mask
