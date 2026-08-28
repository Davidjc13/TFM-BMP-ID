"""Banco de tensores NYIT por usuario (sin IMU, sin texto)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bmpid.config import REPO_ROOT
from bmpid.features.normalize import apply_zscore, fit_zscore
from bmpid.privacy.asserts import assert_no_raw_text

CORE = ("keystroke", "signature", "scroll")
FILE_FOR_TOKENS = {"keystroke": "keystroke.npz", "signature": "signature.npz", "scroll": "scroll_tokens.npz"}
FILE_FOR_VECTOR = {"keystroke": "keystroke.npz", "signature": "signature.npz", "scroll": "scroll.npz"}
ZKEY_TOKENS = {"keystroke": "keystroke", "signature": "signature", "scroll": "scroll_tokens"}
ZKEY_VECTOR = {"keystroke": "keystroke", "signature": "signature", "scroll": "scroll"}


def features_dir(cfg: dict[str, Any]) -> Path:
    custom = (cfg.get("paths") or {}).get("features_dir")
    if custom:
        raw = Path(custom)
        return raw if raw.is_absolute() else REPO_ROOT / raw
    name = str((cfg.get("dataset") or {}).get("name", "nyit"))
    out = Path((cfg.get("paths") or {}).get("processed_root", "data/processed"))
    if not out.is_absolute():
        out = REPO_ROOT / out
    return out / name / "features"


def load_norm_stats(feat_dir: Path) -> dict[str, Any]:
    path = feat_dir / "norm_stats.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert_no_raw_text(data)
    return data


class NyitModalityBank:
    """Índices por split/usuario de una modalidad ya clippeada."""

    def __init__(self, x: np.ndarray, mask: np.ndarray, users: np.ndarray, splits: np.ndarray) -> None:
        self.x = x.astype(np.float32, copy=False)
        self.mask = mask.astype(bool, copy=False)
        self.users = users.astype(str)
        self.splits = splits.astype(str)
        self.by_split_user: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for sp in np.unique(self.splits):
            sp_i = np.where(self.splits == sp)[0]
            for uid in np.unique(self.users[sp_i]):
                self.by_split_user[str(sp)][str(uid)] = sp_i[self.users[sp_i] == uid]

    def users_in(self, split: str) -> list[str]:
        return sorted(self.by_split_user.get(split, {}))

    def indices(self, split: str, user_id: str) -> np.ndarray:
        return self.by_split_user.get(split, {}).get(user_id, np.zeros((0,), dtype=int))


class NyitFeatureStore:
    def __init__(
        self,
        cfg: dict[str, Any],
        modalities: Sequence[str] = CORE,
        *,
        scroll_source: str = "tokens",
    ) -> None:
        mods = tuple(m for m in modalities if m in CORE)
        if "imu" in modalities:
            raise ValueError("IMU no entra en el store del sistema propuesto.")
        if scroll_source not in {"tokens", "vector"}:
            raise ValueError("scroll_source debe ser tokens o vector.")
        self.modalities = mods
        self.scroll_source = scroll_source
        self.feat_dir = features_dir(cfg)
        self.stats = load_norm_stats(self.feat_dir)
        files = FILE_FOR_VECTOR if scroll_source == "vector" else FILE_FOR_TOKENS
        self._zkey = ZKEY_VECTOR if scroll_source == "vector" else ZKEY_TOKENS
        self.banks: dict[str, NyitModalityBank] = {}
        for m in mods:
            path = self.feat_dir / files[m]
            payload = np.load(path, allow_pickle=True)
            x = np.asarray(payload["x"])
            if x.ndim == 2:
                x = x[:, None, :]
                mask = np.ones((x.shape[0], 1), dtype=bool)
            else:
                mask = payload["mask"] if "mask" in payload.files else np.ones(x.shape[:-1], dtype=bool)
            self.banks[m] = NyitModalityBank(x, mask, payload["user_id"], payload["split"])

    def users(self, split: str, *, require_all: bool = False) -> list[str]:
        sets = [set(self.banks[m].users_in(split)) for m in self.modalities]
        if not sets:
            return []
        acc = sets[0]
        for s in sets[1:]:
            acc = acc & s if require_all else acc | s
        return sorted(acc)

    def zscore_train(self, modality: str) -> dict[str, list[float]]:
        return self.stats["modalities"][self._zkey[modality]]["zscore_train"]

    def empty_view(self, modality: str) -> tuple[np.ndarray, np.ndarray]:
        bank = self.banks[modality]
        t = bank.x.shape[1]
        c = bank.x.shape[2]
        return np.zeros((t, c), dtype=np.float32), np.zeros((t,), dtype=bool)

    def fetch(
        self,
        modality: str,
        split: str,
        user_id: str,
        idx_in_user: int,
        *,
        zstats: dict[str, list[float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        bank = self.banks[modality]
        ids = bank.indices(split, user_id)
        if ids.size == 0:
            x, m = self.empty_view(modality)
            return x, m, False
        j = int(ids[int(idx_in_user) % int(ids.size)])
        x = bank.x[j].copy()
        mask = bank.mask[j].copy()
        stats = zstats if zstats is not None else self.zscore_train(modality)
        x = apply_zscore(x, stats)
        x = np.where(mask[..., None], x, 0.0).astype(np.float32)
        ok = bool(np.asarray(mask).any()) if modality == "imu" else True
        return x, mask, ok

    def n_items(self, modality: str, split: str, user_id: str) -> int:
        if user_id not in self.banks[modality].by_split_user.get(split, {}):
            return 0
        return int(self.banks[modality].indices(split, user_id).size)

    def enroll_zscore(
        self,
        modality: str,
        split: str,
        user_id: str,
        enroll_idx: Sequence[int],
    ) -> dict[str, list[float]]:
        if self.n_items(modality, split, user_id) == 0:
            return self.zscore_train(modality)
        bank = self.banks[modality]
        ids = bank.indices(split, user_id)
        rows = [int(ids[int(i) % int(ids.size)]) for i in enroll_idx]
        x = bank.x[np.asarray(rows, dtype=int)]
        mask = bank.mask[np.asarray(rows, dtype=int)]
        return fit_zscore(x, mask)
