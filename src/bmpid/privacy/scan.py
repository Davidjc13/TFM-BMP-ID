"""Auditoría de artefactos persistidos: no debe haber campos de texto de teclas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from bmpid.privacy.asserts import forbidden_fields_present

_JSON_OK = {".json", ".jsonl"}
_YAML_OK = {".yaml", ".yml"}
_SKIP_DIR = {".git", "__pycache__", ".pytest_cache"}


def _load(path: Path) -> Any | None:
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix.lower() == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows
        if path.suffix.lower() in _YAML_OK:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.suffix.lower() == ".csv":
            header = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
            if not header:
                return None
            return {"columns": [c.strip() for c in header[0].split(",")]}
        if path.suffix.lower() == ".npz":
            import numpy as np

            with np.load(path, allow_pickle=False) as blob:
                return {"npz_keys": list(blob.files)}
    except (OSError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError):
        return {"_scan_error": path.name}
    return None


def scan_path(path: Path) -> list[str]:
    obj = _load(path)
    if obj is None:
        return []
    return [f"{path}:{h}" for h in forbidden_fields_present(obj)]


def scan_tree(root: Path, *, glob: str = "**/*") -> dict[str, Any]:
    hits: list[str] = []
    n_files = 0
    if not root.exists():
        return {"root": str(root), "n_files": 0, "hits": [], "ok": True}
    for path in sorted(root.glob(glob)):
        if not path.is_file():
            continue
        if any(p in _SKIP_DIR for p in path.parts):
            continue
        if path.suffix.lower() not in _JSON_OK | _YAML_OK | {".csv", ".npz"}:
            continue
        n_files += 1
        hits.extend(scan_path(path))
    return {"root": str(root), "n_files": n_files, "hits": hits, "ok": not hits}
