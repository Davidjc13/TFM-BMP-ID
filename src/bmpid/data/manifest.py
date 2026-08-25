"""Escritura del manifiesto por modalidad (parquet o JSONL)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from bmpid.privacy.asserts import assert_no_raw_text

MANIFEST_COLUMNS = (
    "user_id",
    "session_id",
    "split",
    "modality",
    "path",
    "n_events",
    "location",
    "task",
    "trial_id",
    "pressure_available",
    "ok",
    "n_gestures",
)


def _normalize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        assert_no_raw_text(row)
        rec = {k: row.get(k) for k in MANIFEST_COLUMNS}
        out.append(rec)
    return out


def write_manifest(rows: Sequence[dict[str, Any]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    records = _normalize(rows)
    suffix = dest.suffix.lower()
    if suffix == ".jsonl":
        with dest.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
        return dest
    try:
        import pandas as pd
    except ImportError:
        jsonl = dest.with_suffix(".jsonl")
        return write_manifest(records, jsonl)
    df = pd.DataFrame.from_records(records)
    if suffix == ".parquet":
        df.to_parquet(dest, index=False)
    elif suffix in {".csv", ".txt"}:
        df.to_csv(dest, index=False)
    else:
        df.to_parquet(dest, index=False)
    return dest


def read_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    assert_no_raw_text(rec)
                    rows.append(rec)
        return rows
    import pandas as pd

    df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
    records = df.to_dict(orient="records")
    for rec in records:
        assert_no_raw_text(rec)
    return records
