"""Entrena fusores de score en *dev* (expertos congelados)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bmpid.config import REPO_ROOT, load_config
from bmpid.eval.ensemble import apply_fuser, collect_modality_scores, fuse_and_metrics
from bmpid.eval.metrics import eer_and_threshold
from bmpid.fusion.learned import fit_learned
from bmpid.fusion.meta import fit_meta
from bmpid.fusion.weighted import weights_from_config, weights_inv_eer
from bmpid.models.behave_tf import CORE
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.seed import set_global_seed
from bmpid.train.dataset import NyitFeatureStore
from bmpid.train.experts import _ckpt_dir, _ens_cfg, _results_dir, load_expert
from bmpid.train.transformer import _device


def _missing(cfg: dict[str, Any]) -> str:
    return str((cfg.get("fusion") or {}).get("missing_score", "omit"))


def train_fusers(cfg: dict[str, Any]) -> dict[str, Any]:
    set_global_seed(int(cfg["seed"]))
    tcfg = _ens_cfg(cfg)
    device = _device({"train": {"transformer": tcfg}})
    store = NyitFeatureStore(cfg, modalities=CORE, scroll_source="vector")
    experts = {m: load_expert(cfg, m, device) for m in CORE}
    collected = collect_modality_scores(
        experts,
        store,
        "dev",
        cfg,
        device=device,
        collect_all=True,
        pair="min",
        zscore_impostor="claimed",
        include_ttd=False,
    )
    y = np.asarray(collected["y_true"])
    s = np.asarray(collected["scores_modal"])
    p = np.asarray(collected["present"])
    missing = _missing(cfg)

    uni_eer = {}
    for i, m in enumerate(collected["modalities"]):
        ok = p[:, i]
        if ok.sum() and len(np.unique(y[ok])) > 1:
            uni_eer[m], _ = eer_and_threshold(y[ok], s[ok, i])
        else:
            uni_eer[m] = 0.5

    w_eq = weights_from_config(cfg, collected["modalities"])
    w_inv = weights_inv_eer(uni_eer, collected["modalities"])
    learned = fit_learned(s, p, y, seed=int(cfg["seed"]))
    meta = fit_meta(s, p, y, seed=int(cfg["seed"]))

    candidates = {
        "fixed": {"weights": w_eq.tolist()},
        "fixed_inv_eer": {"weights": w_inv.tolist(), "unimodal_eer_dev": uni_eer},
        "learned": {"model": learned},
        "meta": {"model": meta},
    }
    kind_map = {"fixed": "fixed", "fixed_inv_eer": "fixed", "learned": "learned", "meta": "meta"}
    rows = []
    best_name = "fixed"
    best_eer = 1.0
    for name, spec in candidates.items():
        fused = apply_fuser(collected, kind_map[name], spec, missing)
        eer, _ = eer_and_threshold(y, fused)
        rows.append({"name": name, "dev_eer": eer})
        if eer < best_eer:
            best_eer = eer
            best_name = name

    ckpt_dir = _ckpt_dir(cfg)
    payload = {
        "best": best_name,
        "dev_eer": best_eer,
        "candidates": rows,
        "fixed_weights": w_eq.tolist(),
        "fixed_inv_eer_weights": w_inv.tolist(),
        "unimodal_eer_dev": uni_eer,
        "missing_score": missing,
        "modalities": list(collected["modalities"]),
        "learned_logits": learned.logits.detach().cpu().tolist(),
        "learned_scale": float(learned.scale.detach().cpu()),
        "meta_state_dict": {k: v.cpu() for k, v in meta.state_dict().items()},
    }
    torch.save(payload, ckpt_dir / "fusion.pt")
    slim = {k: v for k, v in payload.items() if k != "meta_state_dict"}
    slim["meta_keys"] = list(payload["meta_state_dict"])
    assert_no_raw_text(slim)
    log_path = _results_dir(cfg) / "fusion_log.json"
    log_path.write_text(json.dumps({**slim, "best_dev_eer": best_eer}, indent=2) + "\n", encoding="utf-8")
    return {"checkpoint": str(ckpt_dir / "fusion.pt"), "best": best_name, "dev_eer": best_eer, "log": str(log_path)}


def load_fusion_spec(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    path = _ckpt_dir(cfg) / "fusion.pt"
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    from bmpid.fusion.learned import LearnedWeights
    from bmpid.fusion.meta import MetaFuser

    learned = LearnedWeights(len(blob["modalities"]))
    with torch.no_grad():
        learned.logits.copy_(torch.tensor(blob["learned_logits"]))
        learned.scale.copy_(torch.tensor(blob["learned_scale"]))
    meta = MetaFuser(len(blob["modalities"]))
    meta.load_state_dict(blob["meta_state_dict"])
    meta.eval()
    learned.eval()
    return {
        "blob": blob,
        "fixed": {"weights": blob["fixed_weights"]},
        "fixed_inv_eer": {"weights": blob["fixed_inv_eer_weights"]},
        "learned": {"model": learned},
        "meta": {"model": meta},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena fusores de score en dev.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(json.dumps(train_fusers(cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
