"""Serialización de filas de dataset sin campos de texto."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bmpid.privacy.asserts import _TEXT_FIELDS, assert_no_raw_text


def drop_text_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in _TEXT_FIELDS:
                continue
            out[k] = drop_text_fields(v)
        return out
    if isinstance(obj, list):
        return [drop_text_fields(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(drop_text_fields(v) for v in obj)
    return obj


def dump_json(path: Path, obj: Any) -> None:
    cleaned = drop_text_fields(obj)
    assert_no_raw_text(cleaned)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
