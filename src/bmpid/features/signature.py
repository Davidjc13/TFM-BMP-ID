"""Firma / proxy táctil: trayectoria remuestreada a T muestras.

NYIT no tiene presión ni firma legal: se usa el swipe como `proxy_touch`.
Canales (C=6): x_rel, y_rel, dx, dy, velocidad, aceleración.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bmpid.features.parse_nyit import parse_xy_series, split_gestures

MS_TO_S = 1e-3


def resample_trace(
    gesture: np.ndarray,
    *,
    fs: float = 100.0,
    t_len: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolar a `fs` Hz y recortar/pad a `t_len`. Máscara 1=muestra real.

    `gesture` es [P, >=3] con (t_ms, x, y, ...).
    """
    g = np.asarray(gesture, dtype=np.float64)
    if g.shape[0] < 2:
        return (
            np.zeros((t_len, 6), dtype=np.float32),
            np.zeros((t_len,), dtype=np.uint8),
        )
    t = (g[:, 0] - g[0, 0]) * MS_TO_S
    duration = float(t[-1])
    if duration <= 0:
        return (
            np.zeros((t_len, 6), dtype=np.float32),
            np.zeros((t_len,), dtype=np.uint8),
        )
    n_fs = max(2, int(round(duration * float(fs))) + 1)
    t_grid = np.linspace(0.0, duration, n_fs)
    x = np.interp(t_grid, t, g[:, 1])
    y = np.interp(t_grid, t, g[:, 2])
    x = x - x[0]
    y = y - y[0]
    dt = t_grid[1] - t_grid[0] if n_fs > 1 else 1.0 / float(fs)
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    speed = np.hypot(dx, dy)
    accel = np.gradient(speed, dt)
    feat_fs = np.stack([x, y, dx, dy, speed, accel], axis=1).astype(np.float32)

    out = np.zeros((t_len, 6), dtype=np.float32)
    mask = np.zeros((t_len,), dtype=np.uint8)
    n_keep = min(t_len, feat_fs.shape[0])
    if feat_fs.shape[0] > t_len:
        idx = np.linspace(0, feat_fs.shape[0] - 1, t_len)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, feat_fs.shape[0] - 1)
        a = (idx - i0).astype(np.float32)[:, None]
        out = (1.0 - a) * feat_fs[i0] + a * feat_fs[i1]
        mask[:] = 1
    else:
        out[:n_keep] = feat_fs[:n_keep]
        mask[:n_keep] = 1
    return out, mask


def traces_from_csv(
    path: Path,
    *,
    gap_ms: float = 300.0,
    fs: float = 100.0,
    t_len: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    series = parse_xy_series(path, n_coords=2)
    gestures = split_gestures(series, gap_ms)
    if not gestures:
        return (
            np.zeros((0, t_len, 6), dtype=np.float32),
            np.zeros((0, t_len), dtype=np.uint8),
        )
    feats = []
    masks = []
    for g in gestures:
        f, m = resample_trace(g, fs=fs, t_len=t_len)
        if int(m.sum()) >= 2:
            feats.append(f)
            masks.append(m)
    if not feats:
        return (
            np.zeros((0, t_len, 6), dtype=np.float32),
            np.zeros((0, t_len), dtype=np.uint8),
        )
    return np.stack(feats, axis=0), np.stack(masks, axis=0)
