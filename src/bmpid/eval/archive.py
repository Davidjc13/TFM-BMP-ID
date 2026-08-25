"""Archivo de trazas de eval/train para análisis y memoria (sin texto de teclas)."""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bmpid.config import REPO_ROOT
from bmpid.privacy.asserts import assert_no_raw_text


def archive_root(cfg: dict[str, Any], *, kind: str = "transformer") -> Path:
    ev = cfg.get("eval") or {}
    key = "ensemble_archive_root" if kind == "ensemble" else "archive_root"
    default = "data/ensemble_run" if kind == "ensemble" else "data/transformer_run"
    raw = Path(ev.get(key, default))
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def write_env(cfg: dict[str, Any]) -> dict[str, Any]:
    env = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": int(cfg.get("seed", 42)),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        env["torch"] = None
    assert_no_raw_text(env)
    return env


def persist_verify(dest: Path, tag: str, payload: dict[str, Any], cfg: dict[str, Any]) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    y = np.asarray(payload["y_true"], dtype=np.int8)
    s = np.asarray(payload["scores"], dtype=np.float32)
    np.savez_compressed(
        dest / f"{tag}_scores.npz",
        y_true=y,
        scores=s,
        probe_user_id=np.asarray(payload.get("probe_user_id", []), dtype="U16"),
        claimed_user_id=np.asarray(payload.get("claimed_user_id", []), dtype="U16"),
    )
    det_path = dest / f"{tag}_det.json"
    det_path.write_text(json.dumps(payload["det"], indent=2) + "\n", encoding="utf-8")
    per_user = dest / f"{tag}_per_user.json"
    per_user.write_text(json.dumps(payload.get("per_user", []), indent=2) + "\n", encoding="utf-8")
    inv = dest / f"{tag}_inventory.json"
    inv.write_text(json.dumps(payload.get("inventory", {}), indent=2) + "\n", encoding="utf-8")
    ttd_path = dest / f"{tag}_ttd.json"
    ttd_path.write_text(json.dumps(payload.get("ttd_raw", {}), indent=2) + "\n", encoding="utf-8")
    skip = {"det", "y_true", "scores", "probe_user_id", "claimed_user_id", "ttd_raw", "inventory", "per_user"}
    summary = {k: v for k, v in payload.items() if k not in skip}
    summary["n_scores"] = int(y.size)
    summary_path = dest / f"{tag}_summary.json"
    assert_no_raw_text(summary)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    cfg_path = dest / "config_snapshot.yaml"
    if not cfg_path.is_file():
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "scores": str(dest / f"{tag}_scores.npz"),
        "det": str(det_path),
        "per_user": str(per_user),
        "inventory": str(inv),
        "ttd": str(ttd_path),
    }


def copy_train_log(cfg: dict[str, Any], dest: Path, *, kind: str = "transformer") -> str | None:
    if kind == "ensemble":
        from bmpid.train.experts import _results_dir as _edir

        src = _edir(cfg) / "train_log.json"
    else:
        from bmpid.train.transformer import _results_dir

        src = _results_dir(cfg) / "train_log.json"
    if not src.is_file():
        return None
    out = dest / "train_log.json"
    shutil.copyfile(src, out)
    return str(out)
