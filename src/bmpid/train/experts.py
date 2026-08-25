"""Entrena expertos unimodales (Triplet). Sin IMU."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from bmpid.config import REPO_ROOT, load_config
from bmpid.eval.protocol import verify_split
from bmpid.logging_privacy import configure_logging, format_safe
from bmpid.losses.triplet import semi_hard_negative, triplet_loss
from bmpid.models.behave_tf import CORE
from bmpid.models.experts import ExpertWrapper, build_expert
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.seed import set_global_seed
from bmpid.train.dataset import NyitFeatureStore
from bmpid.train.sampler import Set2setSampler, move_batch, unpack_set2set
from bmpid.train.transformer import _device

_FORBIDDEN_IMPORTS = {"bmpid.features.imu"}


def _ens_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return ((cfg.get("train") or {}).get("ensemble") or {})


def _ckpt_dir(cfg: dict[str, Any]) -> Path:
    raw = Path((cfg.get("paths") or {}).get("checkpoints_root", "checkpoints"))
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    dest = raw / "ensemble"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _results_dir(cfg: dict[str, Any]) -> Path:
    raw = Path((cfg.get("paths") or {}).get("results_root", "results"))
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    dest = raw / "ensemble"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def overfit_synthetic(cfg: dict[str, Any], *, steps: int = 80) -> dict[str, float]:
    set_global_seed(int(cfg["seed"]))
    device = torch.device("cpu")
    model = build_expert(cfg, "keystroke").to(device)
    opt = AdamW(model.parameters(), lr=3e-3)
    margin = float(cfg.get("triplet_margin", 0.2))
    rng = np.random.default_rng(0)
    losses = []

    def make_user(uid: int, n: int) -> dict[str, torch.Tensor]:
        base = 0.2 if uid == 0 else -0.2
        x = np.zeros((n, 50, 3), dtype=np.float32)
        x[..., 1:] = base + 0.35 * rng.normal(size=(n, 50, 2)).astype(np.float32)
        mask = np.ones((n, 50), dtype=bool)
        return {
            "keystroke": torch.from_numpy(x),
            "keystroke_mask": torch.from_numpy(mask),
            "keystroke_present": torch.ones(n, dtype=torch.bool),
        }

    model.train()
    for _ in range(steps):
        a = make_user(0, 1)
        p = make_user(0, 1)
        n = make_user(1, 1)
        packed = {
            "keystroke": torch.cat([a["keystroke"], p["keystroke"], n["keystroke"]], dim=0).to(device),
            "keystroke_mask": torch.cat([a["keystroke_mask"], p["keystroke_mask"], n["keystroke_mask"]], dim=0).to(device),
            "keystroke_present": torch.cat(
                [a["keystroke_present"], p["keystroke_present"], n["keystroke_present"]], dim=0
            ).to(device),
        }
        emb = model(packed)
        enroll, pos, neg = unpack_set2set(emb, 1, 1)
        loss = triplet_loss(enroll[:, 0], pos, neg, margin=margin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {"loss_first": losses[0], "loss_last": losses[-1], "steps": float(steps)}


def train_one(
    cfg: dict[str, Any],
    modality: str,
    *,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    log = configure_logging()
    set_global_seed(int(cfg["seed"]))
    tcfg = _ens_cfg(cfg)
    device = _device({"train": {"transformer": tcfg}})
    store = NyitFeatureStore(cfg, modalities=(modality,), scroll_source="vector")
    sampler = Set2setSampler(
        store,
        "train",
        k_enroll=1,
        rng=np.random.default_rng(int(cfg["seed"])),
        require_all=False,
    )
    model = build_expert(cfg, modality).to(device)
    opt = AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    epochs = int(max_epochs if max_epochs is not None else tcfg.get("epochs", 12))
    steps = int(tcfg.get("steps_per_epoch", 48))
    bs = int(tcfg.get("batch_size", 8))
    margin = float(cfg.get("triplet_margin", 0.2))
    clip = float(tcfg.get("grad_clip", 1.0))
    eval_every = int(tcfg.get("eval_every", 8))
    max_g = int(tcfg.get("max_genuine_per_user", 16))
    max_i = int(tcfg.get("max_impostor_per_claimed", 16))
    ckpt_dir = _ckpt_dir(cfg)
    best_eer = 1.0
    history: list[dict[str, Any]] = []
    best_path = ckpt_dir / f"{modality}.pt"
    use_semi = bool(tcfg.get("semi_hard", True))

    log.info("train expert %s", format_safe({"modality": modality, "device": str(device), "n_train_users": len(sampler.users)}))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for _ in range(steps):
            pack = move_batch(sampler.sample_pack(bs), device)
            k_enroll = int(pack.pop("k_enroll").item())
            bsz = int(pack.pop("batch_size").item())
            emb = model(pack)
            enroll, pos, neg = unpack_set2set(emb, bsz, k_enroll)
            anchor = enroll[:, 0]
            if use_semi and bsz > 1:
                neg = semi_hard_negative(anchor, pos, anchor, margin=margin)
            loss = triplet_loss(anchor, pos, neg, margin=margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            running += float(loss.item())
        mean_loss = running / max(steps, 1)
        row: dict[str, Any] = {"epoch": epoch, "loss": mean_loss, "modality": modality}
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            dev = verify_split(
                model,
                store,
                "dev",
                cfg,
                device=device,
                max_genuine=max_g,
                max_impostor=max_i,
                include_ttd=False,
                collect_all=False,
                pair="max",
            )
            row["dev_eer"] = dev["eer"]
            row["dev_auc"] = dev["auc"]
            if dev["eer"] <= best_eer:
                best_eer = float(dev["eer"])
                payload = {
                    "epoch": epoch,
                    "dev_eer": best_eer,
                    "modality": modality,
                    "model_cfg": cfg.get("model"),
                    "state_dict": model.state_dict(),
                }
                torch.save(payload, best_path)
        history.append(row)
        log.info("epoch %s", format_safe(row))

    summary = {
        "best_dev_eer": best_eer,
        "checkpoint": str(best_path),
        "modality": modality,
        "history": history,
        "n_train_users": len(sampler.users),
        "device": str(device),
    }
    assert_no_raw_text({k: v for k, v in summary.items() if k != "history"})
    return summary


def train_all(cfg: dict[str, Any], *, max_epochs: int | None = None) -> dict[str, Any]:
    parts = [train_one(cfg, m, max_epochs=max_epochs) for m in CORE]
    out = {"experts": parts, "modalities": list(CORE)}
    dest = _results_dir(cfg) / "train_log.json"
    slim = {
        "modalities": list(CORE),
        "experts": [
            {k: v for k, v in p.items() if k != "history"} | {"history": p["history"]} for p in parts
        ],
    }
    dest.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    out["train_log"] = str(dest)
    return out


def load_expert(cfg: dict[str, Any], modality: str, device: torch.device, ckpt: Path | None = None) -> ExpertWrapper:
    path = ckpt or (_ckpt_dir(cfg) / f"{modality}.pt")
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    model = build_expert(cfg, modality)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model


def assert_no_imu_import(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FORBIDDEN_IMPORTS or alias.name.startswith("bmpid.features.imu"):
                    raise AssertionError("train no debe importar IMU")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module in _FORBIDDEN_IMPORTS or node.module.startswith("bmpid.features.imu"):
                raise AssertionError("train no debe importar IMU")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena expertos ensemble + Triplet (NYIT).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--modality", default=None, help="keystroke|signature|scroll; default las tres.")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.overfit:
        print(json.dumps(overfit_synthetic(cfg), indent=2))
        return 0
    if args.modality:
        summary = train_one(cfg, args.modality, max_epochs=args.epochs)
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
        return 0
    summary = train_all(cfg, max_epochs=args.epochs)
    print(json.dumps({"train_log": summary["train_log"], "best": [e["best_dev_eer"] for e in summary["experts"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
