"""Adaptador del NYIT Smartphone Activity Dataset (home + lab, cleaned)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bmpid.data.io_counts import count_swipe_gestures, count_timestamped_rows

KEYSTROKE_NAME = "keystrokes_data.csv"
SWIPE_NAME = "swipe_data.csv"
ACC_NAME = "accelerometer.csv"
GYRO_NAME = "gyroscope.csv"

_TYPE_TASKS = frozenset({"sit_and_type", "walk_and_type", "typing"})
_SWIPE_TASKS = frozenset({"sit_and_swipe", "walk_and_swipe", "swiping"})


def discover_users(root: Path) -> list[str]:
    users: set[str] = set()
    for loc in ("lab", "home"):
        base = root / loc
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir():
                users.add(child.name)
    return sorted(users)


def iter_trials(root: Path) -> Iterator[dict[str, str]]:
    """Cada trial es un directorio con CSV de sensores / interacción."""
    lab = root / "lab"
    if lab.is_dir():
        for user_dir in sorted(p for p in lab.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
                for task_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
                    for trial_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                        yield {
                            "user_id": user_dir.name,
                            "location": "lab",
                            "session": session_dir.name,
                            "task": task_dir.name,
                            "trial_id": trial_dir.name,
                            "trial_dir": str(trial_dir.relative_to(root).as_posix()),
                        }
    home = root / "home"
    if home.is_dir():
        for user_dir in sorted(p for p in home.iterdir() if p.is_dir()):
            for task_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
                for trial_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                    yield {
                        "user_id": user_dir.name,
                        "location": "home",
                        "session": "home",
                        "task": task_dir.name,
                        "trial_id": trial_dir.name,
                        "trial_dir": str(trial_dir.relative_to(root).as_posix()),
                    }


def _session_id(meta: dict[str, str]) -> str:
    return (
        f"{meta['location']}_{meta['user_id']}_{meta['session']}_"
        f"{meta['task']}_{meta['trial_id']}"
    )


def _row(
    *,
    meta: dict[str, str],
    split: str,
    modality: str,
    rel_path: str,
    n_events: int,
    n_gestures: int | None,
    ok: bool,
    pressure_available: bool,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "user_id": meta["user_id"],
        "session_id": _session_id(meta),
        "split": split,
        "modality": modality,
        "path": rel_path,
        "n_events": int(n_events),
        "location": meta["location"],
        "task": meta["task"],
        "trial_id": meta["trial_id"],
        "pressure_available": bool(pressure_available),
        "ok": bool(ok),
    }
    if n_gestures is not None:
        rec["n_gestures"] = int(n_gestures)
    return rec


def iter_manifest_rows(
    root: Path,
    user_to_split: dict[str, str],
    *,
    include_imu: bool = True,
    swipe_as_signature_proxy: bool = True,
    swipe_gap_ms: float = 300.0,
) -> Iterator[dict[str, Any]]:
    """Filas de manifiesto. No incluye caracteres ni coordenadas."""
    root = root.resolve()
    for meta in iter_trials(root):
        split = user_to_split[meta["user_id"]]
        trial = root / meta["trial_dir"]
        task = meta["task"]

        ks = trial / KEYSTROKE_NAME
        if ks.is_file() and task in _TYPE_TASKS:
            n = count_timestamped_rows(ks)
            yield _row(
                meta=meta,
                split=split,
                modality="keystroke",
                rel_path=str(ks.relative_to(root).as_posix()),
                n_events=n,
                n_gestures=None,
                ok=n > 0,
                pressure_available=False,
            )

        sw = trial / SWIPE_NAME
        if sw.is_file() and task in _SWIPE_TASKS:
            n_pts, n_g = count_swipe_gestures(sw, gap_ms=swipe_gap_ms)
            yield _row(
                meta=meta,
                split=split,
                modality="scroll",
                rel_path=str(sw.relative_to(root).as_posix()),
                n_events=n_pts,
                n_gestures=n_g,
                ok=n_pts > 0,
                pressure_available=False,
            )
            if swipe_as_signature_proxy:
                yield _row(
                    meta=meta,
                    split=split,
                    modality="signature",
                    rel_path=str(sw.relative_to(root).as_posix()),
                    n_events=n_pts,
                    n_gestures=n_g,
                    ok=n_pts > 0,
                    pressure_available=False,
                )

        if include_imu:
            acc = trial / ACC_NAME
            if acc.is_file():
                n = count_timestamped_rows(acc)
                yield _row(
                    meta=meta,
                    split=split,
                    modality="imu_acc",
                    rel_path=str(acc.relative_to(root).as_posix()),
                    n_events=n,
                    n_gestures=None,
                    ok=n > 0,
                    pressure_available=False,
                )
            gyro = trial / GYRO_NAME
            if gyro.is_file():
                n = count_timestamped_rows(gyro)
                yield _row(
                    meta=meta,
                    split=split,
                    modality="imu_gyro",
                    rel_path=str(gyro.relative_to(root).as_posix()),
                    n_events=n,
                    n_gestures=None,
                    ok=n > 0,
                    pressure_available=False,
                )
