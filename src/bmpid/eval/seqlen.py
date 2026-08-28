"""Ablación unimodal de tecleo N ∈ {25, 50, 100} (protocolo leftover, split eval)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from bmpid.config import REPO_ROOT, load_config
from bmpid.eval.archive import persist_verify, write_env
from bmpid.eval.metrics import far_frr_at_threshold
from bmpid.eval.protocol import verify_split
from bmpid.features.extract import extract_keystroke_seq_len
from bmpid.logging_privacy import configure_logging, format_safe
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.seed import set_global_seed
from bmpid.train.dataset import NyitFeatureStore
from bmpid.train.experts import load_expert, train_one
from bmpid.train.transformer import _device


def _n_list(cfg: dict[str, Any]) -> list[int]:
    raw = cfg.get("seq_len_ablation") or [25, 50, 100]
    return [int(x) for x in raw]


def _ablation_feat_dir(n: int) -> Path:
    return REPO_ROOT / "data" / "processed" / "nyit" / "features" / "ablation" / f"n{n}"


def _cfg_for_n(cfg: dict[str, Any], n: int) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["seq_len"] = n
    out["seq_len_keystroke"] = n
    out["keystroke_stride"] = max(1, n // 2)
    paths = dict(out.get("paths") or {})
    paths["features_dir"] = str(_ablation_feat_dir(n))
    paths["checkpoints_root"] = str(REPO_ROOT / "checkpoints" / "eval_seqlen" / f"n{n}")
    paths["results_root"] = str(REPO_ROOT / "results" / "eval_seqlen" / f"n{n}")
    out["paths"] = paths
    return out


def _reuse_official_n50(cfg: dict[str, Any]) -> None:
    """N=50 oficial: mismos tensores y checkpoint de ensemble, sin reentrenar."""
    dest = _ablation_feat_dir(50)
    dest.mkdir(parents=True, exist_ok=True)
    src = REPO_ROOT / "data" / "processed" / "nyit" / "features"
    for name in ("keystroke.npz", "keystroke_index.jsonl"):
        a, b = src / name, dest / name
        if a.is_file() and not b.is_file():
            b.write_bytes(a.read_bytes())
    stats = json.loads((src / "norm_stats.json").read_text(encoding="utf-8"))
    slim = {
        "clip_percentiles": stats.get("clip_percentiles"),
        "seq_len": 50,
        "keystroke_stride": stats.get("keystroke_stride"),
        "press_only_keystroke": True,
        "imu_in_dump": False,
        "modalities": {"keystroke": stats["modalities"]["keystroke"]},
    }
    assert_no_raw_text(slim)
    (dest / "norm_stats.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    ck_src = REPO_ROOT / "checkpoints" / "ensemble" / "keystroke.pt"
    ck_dst_dir = REPO_ROOT / "checkpoints" / "eval_seqlen" / "n50" / "ensemble"
    ck_dst_dir.mkdir(parents=True, exist_ok=True)
    ck_dst = ck_dst_dir / "keystroke.pt"
    if ck_src.is_file() and not ck_dst.is_file():
        ck_dst.write_bytes(ck_src.read_bytes())


def run_seqlen(cfg: dict[str, Any] | None = None, *, train: bool = True) -> dict[str, Any]:
    log = configure_logging()
    cfg = cfg or load_config()
    set_global_seed(int(cfg["seed"]))
    root = REPO_ROOT / "data" / "eval_run" / "seqlen"
    root.mkdir(parents=True, exist_ok=True)
    env = write_env(cfg)
    (root / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    index: dict[str, Any] = {"env": env, "runs": []}

    for n in _n_list(cfg):
        hop = max(1, n // 2)
        if n == 50:
            _reuse_official_n50(cfg)
            feat_summary = {"reused": True, "seq_len": 50, "out_dir": str(_ablation_feat_dir(50))}
        else:
            feat_summary = extract_keystroke_seq_len(cfg, n=n, stride=hop, out_dir=_ablation_feat_dir(n))
        ncfg = _cfg_for_n(cfg, n)
        if train and n != 50:
            train_one(ncfg, "keystroke")
        elif n == 50:
            _reuse_official_n50(cfg)
        device = _device({"train": {"transformer": ncfg.get("train", {}).get("ensemble") or {}}})
        store = NyitFeatureStore(ncfg, modalities=("keystroke",), scroll_source="vector")
        model = load_expert(ncfg, "keystroke", device)
        metrics = verify_split(
            model,
            store,
            "eval",
            ncfg,
            device=device,
            collect_all=True,
            pair="min",
            zscore_impostor="owner",
            include_ttd=True,
            embed_chunk=int((ncfg.get("eval") or {}).get("embed_batch_size", 32)),
        )
        out = {
            "model": "ensemble_keystroke",
            "seq_len": n,
            "stride": hop,
            "seed": int(cfg["seed"]),
            **metrics,
        }
        dev = verify_split(
            model,
            store,
            "dev",
            ncfg,
            device=device,
            collect_all=True,
            pair="min",
            zscore_impostor="owner",
            include_ttd=False,
            embed_chunk=int((ncfg.get("eval") or {}).get("embed_batch_size", 32)),
        )
        tau = float(dev["threshold_split_eer"])
        out["operating_point_dev_tau"] = far_frr_at_threshold(
            np.asarray(metrics["y_true"]),
            np.asarray(metrics["scores"]),
            tau,
        )
        dest = root / f"n{n}"
        persist_verify(dest, "eval", out, ncfg)
        slim = {k: v for k, v in out.items() if k not in {"det", "y_true", "scores", "probe_user_id", "claimed_user_id", "ttd_raw", "inventory", "per_user"}}
        (dest / "eval_summary.json").write_text(json.dumps({**slim, "n_scores": len(out["y_true"])}, indent=2) + "\n", encoding="utf-8")
        (dest / "dev_summary.json").write_text(
            json.dumps(
                {k: v for k, v in dev.items() if k not in {"det", "y_true", "scores", "probe_user_id", "claimed_user_id", "ttd_raw", "inventory", "per_user"}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        persist_verify(dest, "dev", {**dev, "model": "ensemble_keystroke", "seq_len": n}, ncfg)
        row = {
            "seq_len": n,
            "stride": hop,
            "eer": out["eer"],
            "auc": out["auc"],
            "n_genuine": out["n_genuine"],
            "n_impostor": out["n_impostor"],
            "features": feat_summary,
        }
        index["runs"].append(row)
        log.info("seqlen %s", format_safe(row))

    assert_no_raw_text({k: v for k, v in index.items() if k != "runs"})
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index
