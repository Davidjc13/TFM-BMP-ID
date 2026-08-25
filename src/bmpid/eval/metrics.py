"""EER, AUC, FAR@FRR y TTD (protocolo de verificación, no softmax)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from bmpid.privacy.asserts import assert_no_raw_text


def eer_and_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """EER y umbral (score ≥ τ → genuino). scores mayores = más genuinos."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    tau = float(thr[i])
    return eer, tau


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def far_at_frr(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    target_frr: float = 0.01,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1.0 - tpr
    ok = np.where(fnr <= target_frr)[0]
    if ok.size == 0:
        i = int(np.argmin(np.abs(fnr - target_frr)))
    else:
        i = int(ok[np.argmin(fpr[ok])])
    return {
        "far": float(fpr[i]),
        "frr": float(fnr[i]),
        "threshold": float(thr[i]),
        "target_frr": float(target_frr),
    }


def far_frr_at_threshold(y_true: np.ndarray, scores: np.ndarray, tau: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    pred = s >= float(tau)
    gen = y == 1
    imp = y == 0
    frr = float(1.0 - pred[gen].mean()) if gen.any() else float("nan")
    far = float(pred[imp].mean()) if imp.any() else float("nan")
    return {"tau": float(tau), "far": far, "frr": frr}


def det_curve(y_true: np.ndarray, scores: np.ndarray) -> dict[str, list[float]]:
    fpr, tpr, thr = roc_curve(np.asarray(y_true, dtype=int), np.asarray(scores, dtype=np.float64))
    fnr = (1.0 - tpr).tolist()
    payload = {
        "far": fpr.astype(float).tolist(),
        "frr": fnr,
        "threshold": thr.astype(float).tolist(),
    }
    assert_no_raw_text(payload)
    return payload


def time_to_detect(
    scores: np.ndarray,
    tau: float,
    *,
    k_consecutive: int = 2,
) -> int | None:
    """Índice 1-based de la ventana que dispara (k scores < τ seguidos). None si no hay detección."""
    run = 0
    for i, sc in enumerate(np.asarray(scores, dtype=np.float64)):
        if sc < tau:
            run += 1
            if run >= int(k_consecutive):
                return i + 1
        else:
            run = 0
    return None


def summarize_ttd(values: list[int | None]) -> dict[str, Any]:
    detected = [int(v) for v in values if v is not None]
    n = len(values)
    out = {
        "n_sequences": n,
        "n_detected": len(detected),
        "detect_rate": float(len(detected) / n) if n else float("nan"),
        "mean_events": float(np.mean(detected)) if detected else float("nan"),
        "median_events": float(np.median(detected)) if detected else float("nan"),
    }
    assert_no_raw_text(out)
    return out
