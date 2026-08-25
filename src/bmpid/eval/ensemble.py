"""Eval de expertos + fusión de scores (mismo protocolo leftover que el Transformer)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from bmpid.eval.metrics import (
    auc_score,
    det_curve,
    eer_and_threshold,
    far_at_frr,
    summarize_ttd,
    time_to_detect,
)
from bmpid.eval.protocol import (
    _enroll_indices,
    _query_indices,
    _zstats_enroll,
    embed_views,
    inventory_split,
    leftover_indices,
)
from bmpid.fusion.learned import LearnedWeights
from bmpid.fusion.meta import MetaFuser
from bmpid.fusion.weighted import fuse_fixed
from bmpid.losses.set2set import similarity_score
from bmpid.models.behave_tf import CORE
from bmpid.models.experts import ExpertWrapper
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.train.dataset import NyitFeatureStore


@torch.no_grad()
def collect_modality_scores(
    experts: dict[str, ExpertWrapper],
    store: NyitFeatureStore,
    split: str,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    drop: set[str] | None = None,
    collect_all: bool = True,
    pair: str = "min",
    zscore_impostor: str = "claimed",
    max_genuine: int = 0,
    max_impostor: int = 0,
    include_ttd: bool = True,
) -> dict[str, Any]:
    """Pares genuino/impostor con un score por modalidad y máscara de presencia."""
    k = int(cfg.get("k_enroll", 5))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    ttd_cfg = cfg.get("ttd") or {}
    k_cons = int(ttd_cfg.get("k_consecutive", 2))
    drop = drop or set()
    cap_g = 0 if collect_all else int(max_genuine)
    cap_i = 0 if collect_all else int(max_impostor)
    mods = tuple(m for m in store.modalities if m not in drop)
    users = store.users(split, require_all=False)

    templates: dict[str, dict[str, Tensor]] = {}
    enroll_meta: dict[str, dict[str, Any]] = {}
    leftover: dict[str, dict[str, list[int]]] = {}
    for uid in users:
        eidx = _enroll_indices(store, split, uid, k)
        active = [m for m in mods if len(eidx[m]) > 0]
        if not active:
            continue
        zstats = _zstats_enroll(store, split, uid, eidx)
        leftover[uid] = leftover_indices(store, split, uid, k)
        per: dict[str, Tensor] = {}
        for m in mods:
            if not eidx[m]:
                continue
            emb = embed_views(
                experts[m],
                store,
                split,
                uid,
                {mm: (eidx[m] if mm == m else []) for mm in store.modalities},
                zstats=zstats,
                device=device,
                drop=set(store.modalities) - {m},
            )
            if emb.size(0):
                per[m] = emb
        if not per:
            continue
        templates[uid] = per
        enroll_meta[uid] = {"idx": eidx, "zstats": zstats}

    claimed_users = list(templates)
    labels: list[int] = []
    fused_rows: list[np.ndarray] = []
    present_rows: list[np.ndarray] = []
    probe_uid: list[str] = []
    claimed_uid: list[str] = []
    ttd_seqs: list[np.ndarray] = []
    ttd_pairs: list[dict[str, str]] = []
    per_user: list[dict[str, Any]] = []

    def pair_scores(victim: str, probe: str, zstats: dict[str, dict[str, list[float]]]) -> tuple[np.ndarray, np.ndarray]:
        qidx = _query_indices(leftover[probe], drop, pair=pair, max_q=cap_g, rng=rng)
        n = max((len(qidx[m]) for m in mods), default=0)
        if n == 0:
            return np.zeros((0, len(mods))), np.zeros((0, len(mods)), dtype=bool)
        scores = np.zeros((n, len(mods)), dtype=np.float64)
        present = np.zeros((n, len(mods)), dtype=bool)
        for j, m in enumerate(mods):
            if m not in templates[victim]:
                continue
            n_m = len(qidx.get(m, []))
            if n_m == 0:
                continue
            qemb = embed_views(
                experts[m],
                store,
                split,
                probe,
                {mm: (qidx[m] if mm == m else []) for mm in store.modalities},
                zstats=zstats if zscore_impostor == "claimed" else enroll_meta[probe]["zstats"],
                device=device,
                drop=set(store.modalities) - {m},
            )
            if qemb.size(0) == 0:
                continue
            center = templates[victim][m]
            sc = similarity_score(center.unsqueeze(0).expand(qemb.size(0), -1, -1), qemb)
            take = min(n, int(sc.numel()), n_m)
            scores[:take, j] = sc.detach().cpu().numpy()[:take]
            present[:take, j] = True
        keep = present.any(axis=1)
        return scores[keep], present[keep]

    for victim in claimed_users:
        zstats = enroll_meta[victim]["zstats"]
        s_g, p_g = pair_scores(victim, victim, zstats)
        n_gen = int(s_g.shape[0])
        if n_gen:
            fused_rows.append(s_g)
            present_rows.append(p_g)
            labels.extend([1] * n_gen)
            probe_uid.extend([victim] * n_gen)
            claimed_uid.extend([victim] * n_gen)

        impostors = [u for u in claimed_users if u != victim]
        n_imp_user = 0
        for other in impostors:
            if cap_i > 0 and n_imp_user >= cap_i:
                break
            s_i, p_i = pair_scores(victim, other, zstats)
            if s_i.shape[0] == 0:
                continue
            if cap_i > 0:
                take = min(s_i.shape[0], cap_i - n_imp_user)
                s_i, p_i = s_i[:take], p_i[:take]
            n = int(s_i.shape[0])
            fused_rows.append(s_i)
            present_rows.append(p_i)
            labels.extend([0] * n)
            probe_uid.extend([other] * n)
            claimed_uid.extend([victim] * n)
            n_imp_user += n

        per_user.append(
            {
                "user_id": victim,
                "n_genuine": n_gen,
                "n_impostor": n_imp_user,
                "leftover": {m: len(leftover[victim][m]) for m in store.modalities},
            }
        )

        if include_ttd and "keystroke" in mods and "keystroke" in experts:
            for other in impostors:
                n_oks = store.n_items("keystroke", split, other)
                if n_oks <= k + k_cons:
                    continue
                seq = {m: [] for m in store.modalities}
                seq["keystroke"] = leftover[other]["keystroke"]
                qemb = embed_views(
                    experts["keystroke"],
                    store,
                    split,
                    other,
                    seq,
                    zstats=zstats if zscore_impostor == "claimed" else enroll_meta[other]["zstats"],
                    device=device,
                    drop=set(store.modalities) - {"keystroke"},
                )
                if qemb.size(0) == 0 or "keystroke" not in templates[victim]:
                    continue
                sc = similarity_score(
                    templates[victim]["keystroke"].unsqueeze(0).expand(qemb.size(0), -1, -1),
                    qemb,
                )
                ttd_seqs.append(sc.detach().cpu().numpy())
                ttd_pairs.append({"claimed": victim, "probe": other})

    if not fused_rows:
        raise RuntimeError(f"No hay pares genuino/impostor en split={split}.")
    scores_m = np.concatenate(fused_rows, axis=0)
    present = np.concatenate(present_rows, axis=0)
    y = np.asarray(labels, dtype=int)
    payload = {
        "split": split,
        "modalities": list(mods),
        "scores_modal": scores_m,
        "present": present,
        "y_true": y,
        "probe_user_id": probe_uid,
        "claimed_user_id": claimed_uid,
        "per_user": per_user,
        "ttd_seqs": ttd_seqs,
        "ttd_pairs": ttd_pairs,
        "k_consecutive": k_cons,
        "inventory": inventory_split(store, split, k),
        "n_users": len(claimed_users),
        "k_enroll": k,
        "pair": pair,
        "collect_all": bool(collect_all),
        "zscore_impostor": zscore_impostor,
        "drop_modalities": sorted(drop),
    }
    assert_no_raw_text({k: v for k, v in payload.items() if k not in {"scores_modal", "present", "y_true", "ttd_seqs"}})
    return payload


def metrics_from_scores(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    ttd_seqs: Sequence[np.ndarray] | None = None,
    k_consecutive: int = 2,
) -> dict[str, Any]:
    eer, tau = eer_and_threshold(y, scores)
    ttd_events = [time_to_detect(seq, tau, k_consecutive=k_consecutive) for seq in (ttd_seqs or [])]
    return {
        "n_genuine": int((y == 1).sum()),
        "n_impostor": int((y == 0).sum()),
        "eer": eer,
        "auc": auc_score(y, scores),
        "threshold_split_eer": tau,
        "far_at_1frr": far_at_frr(y, scores, target_frr=0.01),
        "ttd": summarize_ttd(ttd_events),
        "det": det_curve(y, scores),
        "scores": scores.astype(float).tolist(),
        "y_true": y.astype(int).tolist(),
        "ttd_raw": {
            "events": ttd_events,
            "lengths": [int(np.asarray(a).size) for a in (ttd_seqs or [])],
        },
    }


def fuse_and_metrics(
    collected: dict[str, Any],
    fused: np.ndarray,
    *,
    ttd_fused_seqs: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    y = np.asarray(collected["y_true"], dtype=int)
    extra = metrics_from_scores(
        y,
        fused,
        ttd_seqs=ttd_fused_seqs if ttd_fused_seqs is not None else collected.get("ttd_seqs"),
        k_consecutive=int(collected.get("k_consecutive", 2)),
    )
    ttd_raw = extra.pop("ttd_raw")
    ttd_raw["claimed"] = [p["claimed"] for p in collected.get("ttd_pairs", [])]
    ttd_raw["probe"] = [p["probe"] for p in collected.get("ttd_pairs", [])]
    out = {
        "model": "ensemble",
        "split": collected["split"],
        "modalities": collected["modalities"],
        "n_users": collected["n_users"],
        "drop_modalities": collected["drop_modalities"],
        "k_enroll": collected["k_enroll"],
        "pair": collected["pair"],
        "collect_all": collected["collect_all"],
        "zscore_impostor": collected["zscore_impostor"],
        "inventory": collected["inventory"],
        "per_user": collected["per_user"],
        "probe_user_id": collected["probe_user_id"],
        "claimed_user_id": collected["claimed_user_id"],
        "ttd_raw": ttd_raw,
        **extra,
    }
    assert_no_raw_text({k: v for k, v in out.items() if k not in {"det", "y_true", "scores", "probe_user_id", "claimed_user_id", "ttd_raw", "inventory", "per_user"}})
    return out


def apply_fuser(collected: dict[str, Any], kind: str, spec: dict[str, Any], missing: str) -> np.ndarray:
    s = np.asarray(collected["scores_modal"], dtype=np.float64)
    p = np.asarray(collected["present"], dtype=bool)
    mods = list(collected["modalities"])
    if kind == "fixed":
        w_raw = spec["weights"]
        if isinstance(w_raw, dict):
            w = np.asarray([float(w_raw[m]) for m in mods], dtype=np.float64)
        elif len(w_raw) == len(mods):
            w = np.asarray(w_raw, dtype=np.float64)
        else:
            mapped = dict(zip(CORE, list(w_raw)))
            w = np.asarray([mapped[m] for m in mods], dtype=np.float64)
        w = w / max(float(w.sum()), 1e-8)
        return fuse_fixed(s, p, w, missing=missing)
    if kind == "learned":
        model: LearnedWeights = spec["model"]
        if s.shape[1] != int(model.logits.numel()):
            w = model.weights().detach().cpu().numpy()
            mapped = dict(zip(CORE, list(w)))
            ww = np.asarray([mapped[m] for m in mods], dtype=np.float64)
            return fuse_fixed(s, p, ww, missing=missing)
        with torch.no_grad():
            st = torch.from_numpy(s.astype(np.float32))
            pt = torch.from_numpy(p.astype(np.float32))
            return model.fuse(st, pt).numpy()
    if kind == "meta":
        model = spec["model"]
        if s.shape[1] != 3:
            mapped = dict(zip(CORE, spec.get("fallback_weights", [1.0, 1.0, 1.0])))
            ww = np.asarray([mapped.get(m, 1.0) for m in mods], dtype=np.float64)
            return fuse_fixed(s, p, ww, missing=missing)
        with torch.no_grad():
            st = torch.from_numpy(s.astype(np.float32))
            pt = torch.from_numpy(p.astype(np.float32))
            return model(st, pt).numpy()
    raise ValueError(kind)


def ttd_seqs_on_fused_scale(
    collected: dict[str, Any],
    kind: str,
    spec: dict[str, Any],
    missing: str,
) -> list[np.ndarray]:
    """Proyecta las secuencias TTD (solo tecleo) al mismo espacio que el score fusionado."""
    mods = list(collected["modalities"])
    if "keystroke" not in mods:
        return []
    j = mods.index("keystroke")
    out: list[np.ndarray] = []
    for seq in collected.get("ttd_seqs") or []:
        arr = np.asarray(seq, dtype=np.float64).reshape(-1)
        s = np.zeros((arr.size, len(mods)), dtype=np.float64)
        p = np.zeros((arr.size, len(mods)), dtype=bool)
        s[:, j] = arr
        p[:, j] = True
        fake = {**collected, "scores_modal": s, "present": p, "modalities": mods}
        out.append(apply_fuser(fake, kind, spec, missing))
    return out
