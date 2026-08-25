"""Eval de embeddings: EER / AUC / FAR@1%FRR / TTD y archivo de trazas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bmpid.config import REPO_ROOT, load_config
from bmpid.eval.archive import archive_root, copy_train_log, persist_verify, write_env
from bmpid.eval.metrics import far_frr_at_threshold
from bmpid.eval.protocol import verify_split
from bmpid.logging_privacy import configure_logging, format_safe
from bmpid.models.behave_tf import CORE, BehaveTransformer
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.seed import set_global_seed
from bmpid.train.dataset import NyitFeatureStore
from bmpid.train.transformer import _device, _results_dir, _train_cfg


def _load_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device) -> tuple[BehaveTransformer, list[str]]:
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    mods = tuple(blob.get("modalities") or CORE)
    model = BehaveTransformer.from_config(cfg, modalities=mods)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, list(mods)


def _eval_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("eval") or {}


def evaluate(
    cfg: dict[str, Any],
    *,
    checkpoint: str | Path,
    split: str,
    drop: set[str] | None = None,
    collect_all: bool | None = None,
    pair: str | None = None,
    zscore_impostor: str | None = None,
) -> dict[str, Any]:
    log = configure_logging()
    set_global_seed(int(cfg["seed"]))
    device = _device(cfg)
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    model, mods = _load_model(cfg, ckpt, device)
    store = NyitFeatureStore(cfg, modalities=mods)
    tcfg = _train_cfg(cfg)
    ev = _eval_cfg(cfg)
    collect = bool(ev.get("collect_all_traces", True) if collect_all is None else collect_all)
    pair_mode = str(pair or ev.get("pair", "min"))
    zimp = str(zscore_impostor or ev.get("zscore_impostor", "claimed"))
    metrics = verify_split(
        model,
        store,
        split,
        cfg,
        device=device,
        drop=drop,
        max_genuine=int(tcfg.get("max_genuine_per_user", 0)),
        max_impostor=int(tcfg.get("max_impostor_per_claimed", 0)),
        collect_all=collect,
        pair=pair_mode,
        zscore_impostor=zimp,
        embed_chunk=int(ev.get("embed_batch_size", 32)),
        include_ttd=True,
    )
    out = {
        "model": "transformer",
        "checkpoint": str(ckpt),
        "modalities": mods,
        "seed": int(cfg["seed"]),
        **metrics,
    }
    if split == "eval" and not drop:
        arch = archive_root(cfg)
        tau = None
        for cand in (arch / "fused" / "dev_summary.json", _results_dir(cfg) / "dev.json"):
            if cand.is_file():
                tau = float(json.loads(cand.read_text(encoding="utf-8"))["threshold_split_eer"])
                break
        if tau is not None:
            out["operating_point_dev_tau"] = far_frr_at_threshold(
                np.asarray(metrics["y_true"]),
                np.asarray(metrics["scores"]),
                tau,
            )
    assert_no_raw_text({k: v for k, v in out.items() if k not in {"y_true", "scores", "det"}})
    log.info(
        "eval %s",
        format_safe({k: v for k, v in out.items() if k not in {"det", "y_true", "scores", "probe_user_id", "claimed_user_id", "ttd_raw", "inventory", "per_user"}}),
    )
    return out


def _tag(split: str, drop: set[str], extra: str = "") -> str:
    tag = split
    if drop:
        tag += "_no_" + "_".join(sorted(drop))
    if extra:
        tag += extra
    return tag


def _write_results_and_archive(
    cfg: dict[str, Any],
    out: dict[str, Any],
    drop: set[str],
    subdir: str,
    extra: str = "",
    *,
    kind: str = "transformer",
) -> str:
    if kind == "ensemble":
        from bmpid.train.experts import _results_dir as _edir

        dest = _edir(cfg)
    else:
        dest = _results_dir(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    tag = _tag(out["split"], drop, extra)
    slim = {
        k: v
        for k, v in out.items()
        if k not in {"det", "y_true", "scores", "probe_user_id", "claimed_user_id", "ttd_raw", "inventory", "per_user"}
    }
    (dest / f"{tag}.json").write_text(json.dumps({**slim, "n_scores": len(out["y_true"])}, indent=2) + "\n", encoding="utf-8")
    arch = archive_root(cfg, kind=kind) / subdir
    persist_verify(arch, tag, out, cfg)
    return tag


def run_full_archive(cfg: dict[str, Any], checkpoint: str | Path) -> dict[str, Any]:
    """Dev+eval fusionados (todas las trazas emparejadas) y unimodal con todas las trazas."""
    log = configure_logging()
    root = archive_root(cfg)
    env = write_env(cfg)
    (root / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    copy_train_log(cfg, root, kind="transformer")
    index: dict[str, Any] = {"env": env, "runs": []}

    for split in ("dev", "eval"):
        out = evaluate(cfg, checkpoint=checkpoint, split=split, drop=None, collect_all=True, pair="min", zscore_impostor="claimed")
        tag = _write_results_and_archive(cfg, out, set(), "fused", kind="transformer")
        index["runs"].append({"kind": "fused", "tag": tag, "eer": out["eer"], "n_genuine": out["n_genuine"], "n_impostor": out["n_impostor"]})
        log.info("archived fused %s", format_safe(index["runs"][-1]))

    mods = tuple((cfg.get("modalities") or {}).get("core") or CORE)
    for keep in mods:
        drop = set(mods) - {keep}
        out = evaluate(
            cfg,
            checkpoint=checkpoint,
            split="eval",
            drop=drop,
            collect_all=True,
            pair="min",
            zscore_impostor="owner",
        )
        tag = _write_results_and_archive(cfg, out, drop, "unimodal", extra="_unimodal", kind="transformer")
        index["runs"].append(
            {
                "kind": "unimodal",
                "modality": keep,
                "tag": tag,
                "eer": out["eer"],
                "n_genuine": out["n_genuine"],
                "n_impostor": out["n_impostor"],
                "zscore_impostor": "owner",
            }
        )
        log.info("archived unimodal %s", format_safe(index["runs"][-1]))

    for drop, name in (({"scroll"}, "ks_sig"), ({"signature", "scroll"}, "ks")):
        out = evaluate(
            cfg,
            checkpoint=checkpoint,
            split="eval",
            drop=drop,
            collect_all=True,
            pair="min",
            zscore_impostor="claimed",
        )
        tag = _write_results_and_archive(cfg, out, drop, "fused", kind="transformer")
        index["runs"].append({"kind": name, "tag": tag, "eer": out["eer"], "n_genuine": out["n_genuine"], "n_impostor": out["n_impostor"]})

    assert_no_raw_text(index)
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def _attach_dev_tau(cfg: dict[str, Any], out: dict[str, Any], kind: str) -> None:
    if out.get("split") != "eval":
        return
    from bmpid.train.experts import _results_dir as _edir

    dest = _edir(cfg) if kind == "ensemble" else _results_dir(cfg)
    arch = archive_root(cfg, kind=kind)
    tau = None
    for cand in (arch / "fused" / "dev_summary.json", dest / "dev.json"):
        if cand.is_file():
            tau = float(json.loads(cand.read_text(encoding="utf-8"))["threshold_split_eer"])
            break
    if tau is None:
        return
    out["operating_point_dev_tau"] = far_frr_at_threshold(
        np.asarray(out["y_true"]),
        np.asarray(out["scores"]),
        tau,
    )


def run_ensemble_archive(cfg: dict[str, Any]) -> dict[str, Any]:
    from bmpid.eval.ensemble import apply_fuser, collect_modality_scores, fuse_and_metrics, ttd_seqs_on_fused_scale
    from bmpid.train.experts import _ens_cfg, load_expert
    from bmpid.train.fusion import load_fusion_spec

    log = configure_logging()
    set_global_seed(int(cfg["seed"]))
    tcfg = _ens_cfg(cfg)
    device = _device({"train": {"transformer": tcfg}})
    store = NyitFeatureStore(cfg, modalities=CORE, scroll_source="vector")
    experts = {m: load_expert(cfg, m, device) for m in CORE}
    fspec = load_fusion_spec(cfg, device)
    missing = str((cfg.get("fusion") or {}).get("missing_score", "omit"))
    kind_map = {"fixed": "fixed", "fixed_inv_eer": "fixed", "learned": "learned", "meta": "meta"}
    root = archive_root(cfg, kind="ensemble")
    env = write_env(cfg)
    (root / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    copy_train_log(cfg, root, kind="ensemble")
    index: dict[str, Any] = {"env": env, "runs": [], "best_fuser_dev": fspec["blob"]["best"]}

    collected_cache: dict[str, Any] = {}
    for split in ("dev", "eval"):
        collected_cache[split] = collect_modality_scores(
            experts, store, split, cfg, device=device, collect_all=True, pair="min", zscore_impostor="claimed"
        )
        name = str(fspec["blob"]["best"])
        fused = apply_fuser(collected_cache[split], kind_map[name], fspec[name], missing)
        ttd = ttd_seqs_on_fused_scale(collected_cache[split], kind_map[name], fspec[name], missing)
        out = fuse_and_metrics(collected_cache[split], fused, ttd_fused_seqs=ttd)
        out["fusion"] = name
        _attach_dev_tau(cfg, out, "ensemble")
        tag = _write_results_and_archive(cfg, out, set(), "fused", kind="ensemble")
        index["runs"].append({"kind": "fused", "fusion": name, "tag": tag, "eer": out["eer"], "n_genuine": out["n_genuine"], "n_impostor": out["n_impostor"]})
        log.info("archived ensemble fused %s", format_safe(index["runs"][-1]))

    for name in ("fixed", "fixed_inv_eer", "learned", "meta"):
        fused = apply_fuser(collected_cache["eval"], kind_map[name], fspec[name], missing)
        ttd = ttd_seqs_on_fused_scale(collected_cache["eval"], kind_map[name], fspec[name], missing)
        out = fuse_and_metrics(collected_cache["eval"], fused, ttd_fused_seqs=ttd)
        out["fusion"] = name
        tag = _write_results_and_archive(cfg, out, set(), "fused", extra=f"_{name}", kind="ensemble")
        index["runs"].append({"kind": f"fusion_{name}", "tag": tag, "eer": out["eer"], "n_genuine": out["n_genuine"]})

    for drop, name in (({"scroll"}, "ks_sig"), ({"signature", "scroll"}, "ks")):
        col = collect_modality_scores(
            experts, store, "eval", cfg, device=device, drop=drop, collect_all=True, pair="min", zscore_impostor="claimed"
        )
        fused = apply_fuser(col, "fixed", fspec["fixed"], missing)
        ttd = ttd_seqs_on_fused_scale(col, "fixed", fspec["fixed"], missing)
        out = fuse_and_metrics(col, fused, ttd_fused_seqs=ttd)
        out["fusion"] = "fixed"
        tag = _write_results_and_archive(cfg, out, drop, "fused", kind="ensemble")
        index["runs"].append({"kind": name, "tag": tag, "eer": out["eer"], "n_genuine": out["n_genuine"], "n_impostor": out["n_impostor"]})

    for keep in CORE:
        drop = set(CORE) - {keep}
        col = collect_modality_scores(
            experts,
            store,
            "eval",
            cfg,
            device=device,
            drop=drop,
            collect_all=True,
            pair="min",
            zscore_impostor="owner",
        )
        fused = apply_fuser(col, "fixed", {"weights": [1.0]}, missing)
        ttd = ttd_seqs_on_fused_scale(col, "fixed", {"weights": [1.0]}, missing)
        out = fuse_and_metrics(col, fused, ttd_fused_seqs=ttd)
        out["fusion"] = "unimodal"
        tag = _write_results_and_archive(cfg, out, drop, "unimodal", extra="_unimodal", kind="ensemble")
        index["runs"].append(
            {
                "kind": "unimodal",
                "modality": keep,
                "tag": tag,
                "eer": out["eer"],
                "n_genuine": out["n_genuine"],
                "n_impostor": out["n_impostor"],
                "zscore_impostor": "owner",
            }
        )
        log.info("archived ensemble unimodal %s", format_safe(index["runs"][-1]))

    assert_no_raw_text(index)
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    fusion_src = REPO_ROOT / "results" / "ensemble" / "fusion_log.json"
    if fusion_src.is_file():
        import shutil

        shutil.copyfile(fusion_src, root / "fusion_log.json")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval Transformer o ensemble (EER/AUC/FAR/TTD) y archivo de trazas.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/transformer/best.pt")
    parser.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    parser.add_argument("--drop", default="", help="Modalidades a omitir (p.ej. signature,scroll).")
    parser.add_argument("--archive-all", action="store_true", help="Dev+eval fusionados y unimodal; todas las trazas.")
    parser.add_argument("--model", default="transformer", choices=["transformer", "ensemble"])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.model == "ensemble":
        idx = run_ensemble_archive(cfg)
        print(json.dumps(idx, indent=2))
        print(str(archive_root(cfg, kind="ensemble")))
        return 0
    if args.archive_all:
        idx = run_full_archive(cfg, args.checkpoint)
        print(json.dumps(idx, indent=2))
        print(str(archive_root(cfg)))
        return 0
    drop = {m.strip() for m in args.drop.split(",") if m.strip()}
    out = evaluate(cfg, checkpoint=args.checkpoint, split=args.split, drop=drop or None)
    tag = _write_results_and_archive(cfg, out, drop, "fused")
    slim = {
        k: v
        for k, v in out.items()
        if k not in {"det", "y_true", "scores", "probe_user_id", "claimed_user_id", "ttd_raw", "inventory", "per_user"}
    }
    print(json.dumps(slim, indent=2))
    print(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
