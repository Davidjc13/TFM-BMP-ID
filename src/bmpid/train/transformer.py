"""Entrena BehaveTransformer + Set2set sobre features NYIT (sin IMU)."""

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
from bmpid.losses.set2set import set2set_loss
from bmpid.models.behave_tf import CORE, BehaveTransformer
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.seed import set_global_seed
from bmpid.train.dataset import NyitFeatureStore
from bmpid.train.sampler import Set2setSampler, move_batch, unpack_set2set

# Guardia: este módulo no debe importar features.imu
_FORBIDDEN_IMPORTS = {"bmpid.features.imu"}


def _train_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return ((cfg.get("train") or {}).get("transformer") or {})


def _device(cfg: dict[str, Any]) -> torch.device:
    want = str(_train_cfg(cfg).get("device", "auto"))
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ckpt_dir(cfg: dict[str, Any]) -> Path:
    raw = Path((cfg.get("paths") or {}).get("checkpoints_root", "checkpoints"))
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    dest = raw / "transformer"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _results_dir(cfg: dict[str, Any]) -> Path:
    raw = Path((cfg.get("paths") or {}).get("results_root", "results"))
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    dest = raw / "transformer"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _modalities(cfg: dict[str, Any], override: str | None) -> tuple[str, ...]:
    if override:
        mods = tuple(m.strip() for m in override.split(",") if m.strip())
    else:
        t = _train_cfg(cfg)
        mods = tuple(t.get("modalities") or (cfg.get("modalities") or {}).get("core") or CORE)
    if "imu" in mods:
        raise ValueError("IMU no se entrena en el sistema propuesto.")
    return tuple(m for m in mods if m in CORE)


def overfit_synthetic(cfg: dict[str, Any], *, steps: int = 80) -> dict[str, float]:
    """Criterio de hecho: la loss baja en 2 usuarios sintéticos."""
    set_global_seed(int(cfg["seed"]))
    device = torch.device("cpu")
    model = BehaveTransformer.from_config(cfg, modalities=("keystroke",)).to(device)
    opt = AdamW(model.parameters(), lr=3e-3)
    margin = float(cfg.get("set2set_margin", 0.2))
    k = 3
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
        e0 = make_user(0, k)
        p0 = make_user(0, 1)
        n0 = make_user(1, 1)
        packed = {
            "keystroke": torch.cat([e0["keystroke"], p0["keystroke"], n0["keystroke"]], dim=0).to(device),
            "keystroke_mask": torch.cat([e0["keystroke_mask"], p0["keystroke_mask"], n0["keystroke_mask"]], dim=0).to(device),
            "keystroke_present": torch.cat(
                [e0["keystroke_present"], p0["keystroke_present"], n0["keystroke_present"]], dim=0
            ).to(device),
        }
        emb = model(packed)
        enroll, pos, neg = unpack_set2set(emb, 1, k)
        loss = set2set_loss(enroll, pos, neg, margin=margin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {"loss_first": losses[0], "loss_last": losses[-1], "steps": float(steps)}


def train_loop(cfg: dict[str, Any], *, modalities: tuple[str, ...], max_epochs: int | None = None) -> dict[str, Any]:
    log = configure_logging()
    set_global_seed(int(cfg["seed"]))
    tcfg = _train_cfg(cfg)
    device = _device(cfg)
    store = NyitFeatureStore(cfg, modalities=modalities)
    sampler = Set2setSampler(
        store,
        "train",
        k_enroll=int(cfg.get("k_enroll", 5)),
        rng=np.random.default_rng(int(cfg["seed"])),
        require_all=False,
    )
    model = BehaveTransformer.from_config(cfg, modalities=modalities).to(device)
    opt = AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    epochs = int(max_epochs if max_epochs is not None else tcfg.get("epochs", 20))
    steps = int(tcfg.get("steps_per_epoch", 64))
    bs = int(tcfg.get("batch_size", 8))
    margin = float(cfg.get("set2set_margin", 0.2))
    clip = float(tcfg.get("grad_clip", 1.0))
    eval_every = int(tcfg.get("eval_every", 2))
    max_g = int(tcfg.get("max_genuine_per_user", 32))
    max_i = int(tcfg.get("max_impostor_per_claimed", 32))
    ckpt_dir = _ckpt_dir(cfg)
    best_eer = 1.0
    history: list[dict[str, Any]] = []
    best_path = ckpt_dir / "best.pt"

    log.info("train transformer %s", format_safe({"device": str(device), "modalities": list(modalities), "n_train_users": len(sampler.users)}))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for _ in range(steps):
            pack = move_batch(sampler.sample_pack(bs), device)
            k_enroll = int(pack.pop("k_enroll").item())
            bsz = int(pack.pop("batch_size").item())
            emb = model(pack)
            enroll, pos, neg = unpack_set2set(emb, bsz, k_enroll)
            loss = set2set_loss(enroll, pos, neg, margin=margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            running += float(loss.item())
        mean_loss = running / max(steps, 1)
        row: dict[str, Any] = {"epoch": epoch, "loss": mean_loss}
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
                    "modalities": list(modalities),
                    "model_cfg": cfg.get("model"),
                    "state_dict": model.state_dict(),
                }
                torch.save(payload, best_path)
        history.append(row)
        log.info("epoch %s", format_safe(row))

    summary = {
        "best_dev_eer": best_eer,
        "checkpoint": str(best_path),
        "modalities": list(modalities),
        "history": history,
        "n_train_users": len(sampler.users),
        "device": str(device),
    }
    assert_no_raw_text(summary)
    out = _results_dir(cfg) / "train_log.json"
    out.write_text(json.dumps({k: v for k, v in summary.items() if k != "history"} | {"history": history}, indent=2) + "\n", encoding="utf-8")
    summary["train_log"] = str(out)
    return summary


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
    parser = argparse.ArgumentParser(description="Entrena Transformer + Set2set (NYIT).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--modalities", default=None, help="p.ej. keystroke,signature,scroll")
    parser.add_argument("--overfit", action="store_true", help="Smoke test de 2 usuarios sintéticos.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.overfit:
        stats = overfit_synthetic(cfg)
        print(json.dumps(stats, indent=2))
        return 0
    mods = _modalities(cfg, args.modalities)
    summary = train_loop(cfg, modalities=mods, max_epochs=args.epochs)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
