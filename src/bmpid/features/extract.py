"""Ingesta de features NYIT → npz (sin caracteres). IMU ausente salvo flag."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from bmpid.config import REPO_ROOT, load_config
from bmpid.data.build import _dataset_root, build
from bmpid.data.manifest import read_manifest
from bmpid.features.keystroke import windows_from_csv
from bmpid.features.normalize import apply_clip, fit_clip, fit_zscore, stats_payload
from bmpid.features.scroll import tokens_from_vectors, vectors_from_csv
from bmpid.features.signature import traces_from_csv
from bmpid.logging_privacy import configure_logging, format_safe
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.seed import set_global_seed

_INDEX_KEYS = (
    "user_id",
    "session_id",
    "split",
    "modality",
    "location",
    "task",
    "trial_id",
    "n_items",
    "start",
    "end",
)


def _processed_dir(cfg: dict[str, Any]) -> Path:
    name = str((cfg.get("dataset") or {}).get("name", "nyit"))
    out = Path((cfg.get("paths") or {}).get("processed_root", "data/processed"))
    if not out.is_absolute():
        out = REPO_ROOT / out
    return out / name


def _ensure_manifest(cfg: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    dest = _processed_dir(cfg)
    man = dest / "manifest.parquet"
    if not man.is_file():
        alt = dest / "manifest.jsonl"
        man = alt if alt.is_file() else man
    if not man.is_file():
        summary = build(cfg, write_checksums=False)
        man = Path(summary["manifest"])
    return man, read_manifest(man)


def _append(
    bags: dict[str, list],
    *,
    x: np.ndarray,
    mask: np.ndarray | None,
    row: dict[str, Any],
    modality: str,
) -> None:
    n = int(x.shape[0])
    if n == 0:
        return
    start = bags[modality]["_len"]
    bags[modality]["x"].append(x)
    if mask is not None:
        bags[modality]["mask"].append(mask)
    bags[modality]["_len"] = start + n
    rec = {
        "user_id": str(row["user_id"]),
        "session_id": str(row["session_id"]),
        "split": str(row["split"]),
        "modality": modality,
        "location": str(row.get("location", "")),
        "task": str(row.get("task", "")),
        "trial_id": str(row.get("trial_id", "")),
        "n_items": n,
        "start": start,
        "end": start + n,
    }
    assert_no_raw_text(rec)
    bags[modality]["index"].append(rec)


def _stack_mod(bag: dict[str, Any]) -> dict[str, np.ndarray]:
    x = np.concatenate(bag["x"], axis=0) if bag["x"] else np.zeros((0,), dtype=np.float32)
    out: dict[str, np.ndarray] = {"x": x}
    if bag["mask"]:
        out["mask"] = np.concatenate(bag["mask"], axis=0)
    return out


def extract(cfg: dict[str, Any], *, include_imu: bool = False) -> dict[str, Any]:
    log = configure_logging()
    set_global_seed(int(cfg["seed"]))
    root = _dataset_root(cfg)
    _, rows = _ensure_manifest(cfg)

    n = int(cfg.get("seq_len_keystroke") or cfg.get("seq_len") or 50)
    stride = int(cfg.get("keystroke_stride") or 25)
    sig = cfg.get("signature") or {}
    fs = float(sig.get("fs", 100))
    t_len = int(sig.get("T", 128))
    scroll_cfg = cfg.get("scroll") or {}
    gap_ms = float(scroll_cfg.get("gesture_gap_ms", 300.0))
    pre = cfg.get("preproc") or {}
    clip_lo = float(pre.get("clip_low", 1.0))
    clip_hi = float(pre.get("clip_high", 99.0))
    max_g = int(pre.get("max_gestures", 16))
    write_imu = bool(include_imu or pre.get("write_imu_features", False))

    bags: dict[str, dict[str, Any]] = {
        m: {"x": [], "mask": [], "index": [], "_len": 0}
        for m in ("keystroke", "signature", "scroll", "scroll_tokens")
    }
    if write_imu:
        bags["imu"] = {"x": [], "mask": [], "index": [], "_len": 0}

    seen_sig: set[str] = set()
    for row in rows:
        if not row.get("ok", True):
            continue
        rel = str(row["path"])
        path = root / rel
        if not path.is_file():
            continue
        mod = str(row["modality"])
        try:
            if mod == "keystroke":
                x, mask = windows_from_csv(path, n=n, stride=stride)
                _append(bags, x=x, mask=mask, row=row, modality="keystroke")
            elif mod in {"scroll", "signature"}:
                if rel in seen_sig and mod == "signature":
                    continue
                if mod == "scroll":
                    vec = vectors_from_csv(path, gap_ms=gap_ms)
                    _append(bags, x=vec, mask=None, row=row, modality="scroll")
                    tok, tmask = tokens_from_vectors(vec, max_gestures=max_g)
                    if tok.shape[0]:
                        _append(
                            bags,
                            x=tok[None, ...],
                            mask=tmask[None, ...],
                            row=row,
                            modality="scroll_tokens",
                        )
                if mod == "signature" or (
                    str(sig.get("decision", "proxy_touch")) == "proxy_touch" and mod == "scroll"
                ):
                    if rel not in seen_sig:
                        tr, msk = traces_from_csv(path, gap_ms=gap_ms, fs=fs, t_len=t_len)
                        _append(bags, x=tr, mask=msk, row=row, modality="signature")
                        seen_sig.add(rel)
            elif write_imu and mod == "imu_acc":
                from bmpid.features.imu import from_csv_pair

                gyro = path.with_name("gyroscope.csv")
                feat = from_csv_pair(path, gyro if gyro.is_file() else None)
                _append(bags, x=feat, mask=None, row=row, modality="imu")
        except (ValueError, OSError, IndexError):
            continue

    stacked = {m: _stack_mod(b) for m, b in bags.items()}
    train_idx = {}
    for m, bag in bags.items():
        flags: list[bool] = []
        for rec in bag["index"]:
            flags.extend([rec["split"] == "train"] * int(rec["n_items"]))
        train_idx[m] = np.asarray(flags, dtype=bool)

    stats: dict[str, Any] = {
        "clip_percentiles": [clip_lo, clip_hi],
        "seq_len": n,
        "keystroke_stride": stride,
        "signature_fs": fs,
        "signature_T": t_len,
        "press_only_keystroke": True,
        "imu_in_dump": write_imu,
        "modalities": {},
    }

    for m, data in stacked.items():
        x = data["x"]
        if x.size == 0:
            continue
        mask = data.get("mask")
        tr = train_idx[m]
        if mask is not None:
            train_x = x[tr] if tr.any() else x[:0]
            train_m = mask[tr] if tr.any() else mask[:0]
        else:
            train_x = x[tr] if tr.any() else x[:0]
            train_m = None
        c = int(x.shape[-1])
        if train_x.size == 0:
            # No usar eval/dev para clip ni z-score (leakage).
            bounds = {"low": [float("-1e9")] * c, "high": [float("1e9")] * c}
            z = {"mean": [0.0] * c, "std": [1.0] * c}
        else:
            bounds = fit_clip(train_x, low=clip_lo, high=clip_hi, mask=train_m)
            z = fit_zscore(apply_clip(train_x, bounds), train_m)
        x_clip = apply_clip(x, bounds)
        data["x"] = x_clip
        user_z: dict[str, Any] = {}
        by_user: dict[str, list[int]] = defaultdict(list)
        for i, rec in enumerate(bags[m]["index"]):
            if rec["split"] != "train":
                continue
            for k in range(rec["start"], rec["end"]):
                by_user[rec["user_id"]].append(k)
        for uid, idxs in by_user.items():
            sl = x_clip[np.array(idxs, dtype=int)]
            sl_m = mask[np.array(idxs, dtype=int)] if mask is not None else None
            user_z[uid] = fit_zscore(sl, sl_m)
        stats["modalities"][m] = {
            "clip": bounds,
            "zscore_train": z,
            "zscore_user_train": user_z,
            "n_items": int(x_clip.shape[0]),
            "shape": list(x_clip.shape),
        }

    stats = stats_payload(stats)
    out_dir = _processed_dir(cfg) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for m, data in stacked.items():
        if data["x"].size == 0:
            continue
        payload = {k: data[k] for k in data}
        users = []
        splits = []
        for rec in bags[m]["index"]:
            users.extend([rec["user_id"]] * rec["n_items"])
            splits.extend([rec["split"]] * rec["n_items"])
        payload["user_id"] = np.asarray(users)
        payload["split"] = np.asarray(splits)
        dest = out_dir / f"{m}.npz"
        np.savez_compressed(dest, **payload)
        written[m] = str(dest)
        idx_path = out_dir / f"{m}_index.jsonl"
        with idx_path.open("w", encoding="utf-8") as fh:
            for rec in bags[m]["index"]:
                rec_s = {k: rec[k] for k in _INDEX_KEYS}
                assert_no_raw_text(rec_s)
                fh.write(json.dumps(rec_s, ensure_ascii=True) + "\n")

    stats_path = out_dir / "norm_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "out_dir": str(out_dir),
        "files": written,
        "norm_stats": str(stats_path),
        "counts": {m: stats["modalities"][m]["n_items"] for m in stats.get("modalities", {})},
        "imu_written": write_imu,
    }
    log.info("features %s", format_safe(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extrae features NYIT (Fase 2).")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--include-imu",
        action="store_true",
        help="Escribe tensores IMU (no forma parte del sistema propuesto).",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    summary = extract(cfg, include_imu=args.include_imu)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
