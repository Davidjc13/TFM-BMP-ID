"""Recalcula EER/AUC/DET de ensemble y RF sobre el leftover del Transformer 3-mod.

Sin reentrenar: recorta el prefijo de scores por par (claimed, probe) usando
n_genuine del Transformer (tope de secuencias de scroll).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bmpid.config import REPO_ROOT
from bmpid.eval.archive import write_det_csv
from bmpid.eval.metrics import auc_score, det_curve, eer_and_threshold, far_at_frr
from bmpid.privacy.asserts import assert_no_raw_text

TF_PER_USER = REPO_ROOT / "data" / "transformer_run" / "fused" / "eval_per_user.json"
TF_SCORES = REPO_ROOT / "data" / "transformer_run" / "fused" / "eval_scores.npz"
OUT_DIR = REPO_ROOT / "data" / "eval_run"
DET_DIR = OUT_DIR / "det"

RUNS: dict[str, Path] = {
    "transformer_3mod": TF_SCORES,
    "ensemble_meta_3mod": REPO_ROOT / "data" / "ensemble_run" / "fused" / "eval_scores.npz",
    "rf_keystroke": REPO_ROOT / "data" / "rf_run" / "rf_keystroke" / "eval_scores.npz",
    "rf_3mod_features": REPO_ROOT / "data" / "rf_run" / "rf_3mod_features" / "eval_scores.npz",
    "rf_meta_3mod": REPO_ROOT / "data" / "rf_run" / "rf_meta_3mod" / "eval_scores.npz",
}


def leftover_caps(per_user_path: Path = TF_PER_USER) -> dict[str, int]:
    rows = json.loads(per_user_path.read_text(encoding="utf-8"))
    return {str(r["user_id"]): int(r["n_genuine"]) for r in rows}


def match_prefix(
    y: np.ndarray,
    scores: np.ndarray,
    claimed: np.ndarray,
    probe: np.ndarray,
    caps: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Por cada bloque consecutivo (claimed, probe) toma las primeras caps[probe] filas."""
    n = int(y.size)
    keep: list[int] = []
    i = 0
    while i < n:
        c = str(claimed[i])
        p = str(probe[i])
        j = i + 1
        while j < n and str(claimed[j]) == c and str(probe[j]) == p:
            j += 1
        take = min(j - i, int(caps.get(p, 0)))
        keep.extend(range(i, i + take))
        i = j
    idx = np.asarray(keep, dtype=int)
    return y[idx], scores[idx]


def summarize(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    eer, tau = eer_and_threshold(y, scores)
    det = det_curve(y, scores)
    far = far_at_frr(y, scores, target_frr=0.01)
    return {
        "n_genuine": int((y == 1).sum()),
        "n_impostor": int((y == 0).sum()),
        "eer": float(eer),
        "auc": float(auc_score(y, scores)),
        "threshold_split_eer": float(tau),
        "far_at_1frr": far,
        "det": det,
    }


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    y = np.asarray(z["y_true"], dtype=int)
    s = np.asarray(z["scores"], dtype=np.float64)
    claimed = np.asarray(z["claimed_user_id"]).astype(str)
    probe = np.asarray(z["probe_user_id"]).astype(str)
    return y, s, claimed, probe


def run_all(*, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    caps = leftover_caps()
    det_dir = out_dir / "det"
    det_dir.mkdir(parents=True, exist_ok=True)
    systems: dict[str, Any] = {}
    for name, path in RUNS.items():
        y, s, claimed, probe = load_npz(path)
        if name == "transformer_3mod":
            ym, sm = y, s
        else:
            ym, sm = match_prefix(y, s, claimed, probe, caps)
        row = summarize(ym, sm)
        stem = f"matched_{name}_eval"
        write_det_csv(row["det"], det_dir / f"{stem}_det.csv")
        slim = {k: v for k, v in row.items() if k != "det"}
        slim["system"] = name
        slim["source"] = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        slim["matched_to"] = "transformer_3mod leftover prefix"
        systems[name] = slim
    payload = {
        "n_users": len(caps),
        "caps_genuine": caps,
        "note": (
            "Prefijo leftover por usuario: mismas ventanas de tecleo que el Transformer 3-mod "
            "(tope de secuencias de scroll). El canal swipe/scroll no es el mismo objeto físico "
            "entre pipelines. Sin reentrenamiento."
        ),
        "systems": systems,
    }
    assert_no_raw_text(payload)
    dest = out_dir / "matched_subset.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["path"] = str(dest)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Métricas emparejadas Transformer/ensemble/RF.")
    parser.parse_args(argv)
    out = run_all()
    slim = {
        "path": out["path"],
        "systems": {k: {kk: vv for kk, vv in v.items() if kk != "far_at_1frr"} for k, v in out["systems"].items()},
    }
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
