from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config inválida: {cfg_path}")
    _validate_privacy(data)
    return data


def _validate_privacy(cfg: dict[str, Any]) -> None:
    privacy = cfg.get("privacy") or {}
    if privacy.get("store_raw_text", False):
        raise ValueError("privacy.store_raw_text debe ser false (no persistir texto).")
    decision = ((cfg.get("signature") or {}).get("decision") or "").strip()
    if decision and decision not in {"proxy_touch", "separate_corpus"}:
        raise ValueError("signature.decision debe ser proxy_touch o separate_corpus.")
