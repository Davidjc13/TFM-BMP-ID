"""Ingesta NYIT → manifiesto + splits + checksums de ficheros núcleo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bmpid.config import REPO_ROOT, load_config
from bmpid.data.adapters import get_adapter
from bmpid.data.checksums import verify_sha256sums, write_sha256sums
from bmpid.data.manifest import write_manifest
from bmpid.data.nyit import discover_users
from bmpid.data.splits import assert_disjoint_user_splits, assign_user_splits
from bmpid.logging_privacy import configure_logging, format_safe
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.seed import set_global_seed


def _dataset_root(cfg: dict[str, Any]) -> Path:
    ds = cfg.get("dataset") or {}
    raw = ds.get("root") or (cfg.get("paths") or {}).get("data_root")
    if not raw:
        raise ValueError("dataset.root no definido")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def build(cfg: dict[str, Any], *, write_checksums: bool = True) -> dict[str, Any]:
    log = configure_logging()
    set_global_seed(int(cfg["seed"]))
    name = str((cfg.get("dataset") or {}).get("name", "nyit"))
    root = _dataset_root(cfg)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset no encontrado: {root}")

    adapter = get_adapter(name)
    users = discover_users(root)
    split_cfg = cfg.get("splits") or {}
    ratios = tuple(float(x) for x in split_cfg.get("ratios", [0.6, 0.2, 0.2]))
    if len(ratios) != 3:
        raise ValueError("splits.ratios debe ser [train, dev, eval]")
    user_to_split = assign_user_splits(users, seed=int(cfg["seed"]), ratios=ratios)  # type: ignore[arg-type]
    assert_disjoint_user_splits(user_to_split)

    sig = cfg.get("signature") or {}
    swipe_as_sig = str(sig.get("decision", "proxy_touch")) == "proxy_touch"
    imu_cfg = cfg.get("imu") or {}
    include_imu = bool(imu_cfg.get("include_in_manifest", True))

    scroll_cfg = cfg.get("scroll") or {}
    swipe_gap_ms = float(scroll_cfg.get("gesture_gap_ms", 300.0))

    rows = list(
        adapter.iter_rows(
            root,
            user_to_split,
            include_imu=include_imu,
            swipe_as_signature_proxy=swipe_as_sig,
            swipe_gap_ms=swipe_gap_ms,
        )
    )
    for rec in rows:
        assert_no_raw_text(rec)

    out_dir = Path((cfg.get("paths") or {}).get("processed_root", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = write_manifest(rows, out_dir / "manifest.parquet")
    splits_path = out_dir / "splits.json"
    payload = {
        "dataset": name,
        "root": str(root),
        "n_users": len(users),
        "n_rows": len(rows),
        "user_to_split": user_to_split,
        "signature_decision": sig.get("decision"),
        "scroll_task": scroll_cfg.get("task"),
        "imu_in_proposed": bool(imu_cfg.get("enabled_in_proposed", False)),
    }
    assert_no_raw_text(payload)
    splits_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_path = None
    if write_checksums:
        rels = sorted({str(r["path"]) for r in rows if r.get("modality") in {"keystroke", "scroll", "signature"}})
        checksum_rel = (cfg.get("dataset") or {}).get("checksums_file", "data/checksums/nyit_core.sha256")
        checksum_path = Path(checksum_rel)
        if not checksum_path.is_absolute():
            checksum_path = REPO_ROOT / checksum_path
        write_sha256sums(root, rels, checksum_path)

    summary = {
        "manifest": str(manifest_path),
        "splits": str(splits_path),
        "checksums": str(checksum_path) if checksum_path else None,
        "n_users": len(users),
        "n_rows": len(rows),
        "n_train": sum(1 for s in user_to_split.values() if s == "train"),
        "n_dev": sum(1 for s in user_to_split.values() if s == "dev"),
        "n_eval": sum(1 for s in user_to_split.values() if s == "eval"),
        "by_modality": _count_mod(rows),
    }
    log.info("build %s", format_safe(summary))
    return summary


def _count_mod(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        m = str(r.get("modality"))
        out[m] = out.get(m, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construye manifiesto y splits (Fase 1).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-checksums", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.verify_checksums:
        root = _dataset_root(cfg)
        checksum_rel = (cfg.get("dataset") or {}).get("checksums_file", "data/checksums/nyit_core.sha256")
        checksum_path = Path(checksum_rel)
        if not checksum_path.is_absolute():
            checksum_path = REPO_ROOT / checksum_path
        bad = verify_sha256sums(root, checksum_path)
        if bad:
            raise SystemExit(f"Checksums inválidos ({len(bad)}): {bad[:5]}")
        print("checksums_ok")
        return 0
    summary = build(cfg, write_checksums=not args.no_checksums)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
