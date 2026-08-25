"""Ventanas de longitud fija N, stride y máscaras de padding."""

from __future__ import annotations

import numpy as np


def sliding_windows(
    seq: np.ndarray,
    *,
    n: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parte `seq` [L, C] en ventanas [W, N, C] y máscara [W, N] (1=válido).

    Si L < N, una ventana con padding 0. Si L > N, sliding windows sin solapar
    el padding (solo pasos reales).
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    if stride < 1:
        raise ValueError("stride debe ser >= 1")
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq[:, None]
    length, channels = seq.shape
    if length == 0:
        return (
            np.zeros((0, n, channels), dtype=np.float32),
            np.zeros((0, n), dtype=np.uint8),
        )
    if length < n:
        windows = np.zeros((1, n, channels), dtype=np.float32)
        mask = np.zeros((1, n), dtype=np.uint8)
        windows[0, :length] = seq
        mask[0, :length] = 1
        return windows, mask

    starts = list(range(0, length - n + 1, stride))
    last = length - n
    if starts[-1] != last:
        starts.append(last)
    w = len(starts)
    windows = np.empty((w, n, channels), dtype=np.float32)
    mask = np.ones((w, n), dtype=np.uint8)
    for i, s in enumerate(starts):
        windows[i] = seq[s : s + n]
    return windows, mask
