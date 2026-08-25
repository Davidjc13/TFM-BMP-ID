"""Clip por percentiles y z-score. Estadísticos solo de train / enrolamiento."""

from __future__ import annotations

from typing import Any

import numpy as np

from bmpid.privacy.asserts import assert_no_raw_text


def fit_clip(
    values: np.ndarray,
    *,
    low: float = 1.0,
    high: float = 99.0,
    mask: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Percentiles por canal. `values` [..., C]."""
    arr, m = _flat_valid(values, mask)
    if arr.size == 0:
        c = int(values.shape[-1]) if values.ndim else 1
        return {"low": [0.0] * c, "high": [1.0] * c}
    lo = np.percentile(arr, low, axis=0)
    hi = np.percentile(arr, high, axis=0)
    hi = np.maximum(hi, lo)
    return {"low": lo.astype(np.float64).tolist(), "high": hi.astype(np.float64).tolist()}


def apply_clip(values: np.ndarray, bounds: dict[str, list[float]]) -> np.ndarray:
    lo = np.asarray(bounds["low"], dtype=np.float32)
    hi = np.asarray(bounds["high"], dtype=np.float32)
    return np.clip(values.astype(np.float32, copy=False), lo, hi)


def fit_zscore(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, list[float]]:
    arr, _ = _flat_valid(values, mask)
    if arr.size == 0:
        c = int(values.shape[-1]) if values.ndim else 1
        return {"mean": [0.0] * c, "std": [1.0] * c}
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float64).tolist(), "std": std.astype(np.float64).tolist()}


def apply_zscore(values: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return (values.astype(np.float32, copy=False) - mean) / std


def _flat_valid(values: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if mask is None:
        return arr.reshape(-1, arr.shape[-1]), None
    m = np.asarray(mask, dtype=bool)
    if m.shape != arr.shape[:-1]:
        raise ValueError(f"mask shape {m.shape} incompatible con {arr.shape}")
    return arr[m], m


def stats_payload(obj: dict[str, Any]) -> dict[str, Any]:
    assert_no_raw_text(obj)
    return obj
