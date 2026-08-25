"""Fusión de scores con pesos fijos. Modalidad ausente: omitir (renormalizar) o cero."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


CORE = ("keystroke", "signature", "scroll")


def weights_from_config(cfg: dict, modalities: Sequence[str] = CORE) -> np.ndarray:
    raw = ((cfg.get("fusion") or {}).get("fixed_weights") or {})
    w = np.asarray([float(raw.get(m, 1.0)) for m in modalities], dtype=np.float64)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(modalities), dtype=np.float64) / max(len(modalities), 1)


def weights_inv_eer(eers: Mapping[str, float], modalities: Sequence[str] = CORE) -> np.ndarray:
    inv = np.asarray([1.0 / max(float(eers.get(m, 0.5)), 1e-6) for m in modalities], dtype=np.float64)
    return inv / inv.sum()


def fuse_fixed(
    scores: np.ndarray,
    present: np.ndarray,
    weights: np.ndarray,
    *,
    missing: str = "omit",
) -> np.ndarray:
    """scores/present [N, M]. missing=omit renormaliza; zero deja s=0 y el peso."""
    s = np.asarray(scores, dtype=np.float64)
    p = np.asarray(present, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    if missing == "zero":
        num = (s * w) * p
        den = np.full(s.shape[0], w.sum())
        return num.sum(axis=1) / np.maximum(den, 1e-8)
    masked = p.astype(np.float64) * w
    den = masked.sum(axis=1, keepdims=True)
    empty = den[:, 0] <= 1e-12
    den = np.maximum(den, 1e-12)
    out = (s * masked).sum(axis=1, keepdims=True) / den
    out = out[:, 0]
    out[empty] = 0.0
    return out
