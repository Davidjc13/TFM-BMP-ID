"""Protocolo open-set: enrolamiento k, todas las consultas residuales, TTD en tecleo."""

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
from bmpid.losses.set2set import similarity_score
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.train.dataset import NyitFeatureStore
from bmpid.train.sampler import move_batch, views_to_batch


@torch.no_grad()
def embed_views(
    model: Any,
    store: NyitFeatureStore,
    split: str,
    user_id: str,
    indices: dict[str, Sequence[int]],
    *,
    zstats: dict[str, dict[str, list[float]]] | None,
    device: torch.device,
    drop: set[str] | None = None,
    chunk: int = 32,
) -> Tensor:
    drop = drop or set()
    lengths = [len(indices.get(m, [])) for m in store.modalities if m not in drop]
    n = max(lengths) if lengths else 0
    if n == 0:
        return torch.zeros((0, model.d_embed), device=device)
    embs: list[Tensor] = []
    for start in range(0, n, int(chunk)):
        views = []
        for i in range(start, min(start + int(chunk), n)):
            view: dict[str, Any] = {}
            for m in store.modalities:
                n_m = store.n_items(m, split, user_id)
                idx_list = list(indices.get(m, []))
                if m in drop or n_m == 0 or not idx_list:
                    x, mask = store.empty_view(m)
                    view[m] = x
                    view[f"{m}_mask"] = mask
                    view[f"{m}_present"] = False
                    continue
                idx = int(idx_list[i % len(idx_list)])
                zs = None if zstats is None else zstats[m]
                x, mask, present = store.fetch(m, split, user_id, idx, zstats=zs)
                view[m] = x
                view[f"{m}_mask"] = mask
                view[f"{m}_present"] = present
            views.append(view)
        batch = move_batch(views_to_batch(views, store.modalities), device)
        embs.append(model(batch))
    return torch.cat(embs, dim=0)


def _enroll_indices(store: NyitFeatureStore, split: str, user_id: str, k: int) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for m in store.modalities:
        n = store.n_items(m, split, user_id)
        out[m] = list(range(min(k, n))) if n else []
    return out


def leftover_indices(store: NyitFeatureStore, split: str, user_id: str, k: int) -> dict[str, list[int]]:
    """Todas las ventanas/trazas posteriores al enrolamiento, sin muestrear."""
    out: dict[str, list[int]] = {}
    for m in store.modalities:
        n = store.n_items(m, split, user_id)
        out[m] = list(range(k, n)) if n > k else []
    return out


def _query_indices(
    leftover: dict[str, list[int]],
    drop: set[str],
    *,
    pair: str,
    max_q: int,
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    """pair=min: tantas vistas como la modalidad más corta (sin reutilizar).
    pair=max: recorre la más larga (ciclo en las cortas).
    max_q<=0: no recortar; si >0, subconjunto determinista tras barajar con seed.
    """
    active = {m: list(idx) for m, idx in leftover.items() if m not in drop and idx}
    if not active:
        return {m: [] for m in leftover}
    if pair == "min":
        n = min(len(v) for v in active.values())
        picked = {m: idx[:n] for m, idx in leftover.items()}
    else:
        picked = {m: list(idx) for m, idx in leftover.items()}
    if max_q > 0:
        n = max((len(v) for v in picked.values()), default=0)
        if n > max_q:
            take = np.sort(rng.choice(n, size=max_q, replace=False))
            picked = {m: [idx[int(i)] for i in take if i < len(idx)] for m, idx in picked.items()}
    for m in leftover:
        if m in drop:
            picked[m] = []
    return picked


def _zstats_enroll(
    store: NyitFeatureStore,
    split: str,
    user_id: str,
    enroll_idx: dict[str, list[int]],
) -> dict[str, dict[str, list[float]]]:
    return {m: store.enroll_zscore(m, split, user_id, enroll_idx[m] or [0]) for m in store.modalities}


def inventory_split(store: NyitFeatureStore, split: str, k: int) -> dict[str, Any]:
    rows = []
    totals = {m: {"n": 0, "enroll": 0, "leftover": 0, "users": 0} for m in store.modalities}
    for uid in store.users(split, require_all=False):
        rec: dict[str, Any] = {"user_id": uid, "split": split}
        for m in store.modalities:
            n = store.n_items(m, split, uid)
            enr = min(k, n)
            left = max(0, n - k)
            rec[m] = {"n": n, "enroll": enr, "leftover": left}
            totals[m]["n"] += n
            totals[m]["enroll"] += enr
            totals[m]["leftover"] += left
            if n:
                totals[m]["users"] += 1
        rec["fused_min_leftover"] = min(
            (rec[m]["leftover"] for m in store.modalities if rec[m]["leftover"] > 0),
            default=0,
        )
        rows.append(rec)
    payload = {"split": split, "k_enroll": k, "modalities": list(store.modalities), "totals": totals, "users": rows}
    assert_no_raw_text(payload)
    return payload


def verify_split(
    model: Any,
    store: NyitFeatureStore,
    split: str,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    drop: set[str] | None = None,
    max_genuine: int = 0,
    max_impostor: int = 0,
    include_ttd: bool = True,
    pair: str = "min",
    collect_all: bool = True,
    embed_chunk: int = 32,
    zscore_impostor: str = "claimed",
) -> dict[str, Any]:
    """Si collect_all, no se muestrean consultas. max_*<=0 = sin techo.
    pair=min usa todas las trazas de la modalidad más corta por usuario (prefijo alineado).
    """
    k = int(cfg.get("k_enroll", 5))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    ttd_cfg = cfg.get("ttd") or {}
    k_cons = int(ttd_cfg.get("k_consecutive", 2))
    drop = drop or set()
    cap_g = 0 if collect_all else int(max_genuine)
    cap_i = 0 if collect_all else int(max_impostor)
    users = store.users(split, require_all=False)
    labels: list[int] = []
    scores: list[float] = []
    probe_uid: list[str] = []
    claimed_uid: list[str] = []
    ttd_seqs: list[np.ndarray] = []
    ttd_pairs: list[dict[str, str]] = []
    per_user: list[dict[str, Any]] = []
    model.eval()

    templates: dict[str, Tensor] = {}
    enroll_meta: dict[str, dict[str, Any]] = {}
    leftover: dict[str, dict[str, list[int]]] = {}
    for uid in users:
        eidx = _enroll_indices(store, split, uid, k)
        active = [m for m in store.modalities if m not in drop and len(eidx[m]) > 0]
        if not active:
            continue
        zstats = _zstats_enroll(store, split, uid, eidx)
        leftover[uid] = leftover_indices(store, split, uid, k)
        emb = embed_views(
            model, store, split, uid, eidx, zstats=zstats, device=device, drop=drop, chunk=embed_chunk
        )
        if emb.size(0) == 0:
            continue
        templates[uid] = emb
        enroll_meta[uid] = {"idx": eidx, "zstats": zstats}

    claimed_users = list(templates)
    query_cache: dict[str, Tensor] = {}

    def queries_for(uid: str, zstats: dict[str, dict[str, list[float]]]) -> Tensor:
        key = uid if zscore_impostor == "owner" else f"{uid}@{id(zstats)}"
        if zscore_impostor == "owner":
            if uid in query_cache:
                return query_cache[uid]
            zs = enroll_meta[uid]["zstats"]
            qidx = _query_indices(leftover[uid], drop, pair=pair, max_q=cap_g, rng=rng)
            qemb = embed_views(
                model, store, split, uid, qidx, zstats=zs, device=device, drop=drop, chunk=embed_chunk
            )
            query_cache[uid] = qemb
            return qemb
        qidx = _query_indices(leftover[uid], drop, pair=pair, max_q=cap_g, rng=rng)
        return embed_views(
            model, store, split, uid, qidx, zstats=zstats, device=device, drop=drop, chunk=embed_chunk
        )

    for victim in claimed_users:
        center = templates[victim]
        zstats = enroll_meta[victim]["zstats"]
        qemb = queries_for(victim, zstats)
        n_gen = int(qemb.size(0))
        if n_gen:
            sc = similarity_score(center.unsqueeze(0).expand(qemb.size(0), -1, -1), qemb)
            scores.extend(sc.detach().cpu().numpy().tolist())
            labels.extend([1] * n_gen)
            probe_uid.extend([victim] * n_gen)
            claimed_uid.extend([victim] * n_gen)

        impostors = [u for u in claimed_users if u != victim]
        n_imp_user = 0
        for other in impostors:
            if cap_i > 0 and n_imp_user >= cap_i:
                break
            qemb = queries_for(other, zstats)
            if qemb.size(0) == 0:
                continue
            if cap_i > 0:
                take = min(qemb.size(0), cap_i - n_imp_user)
                qemb = qemb[:take]
            sc = similarity_score(center.unsqueeze(0).expand(qemb.size(0), -1, -1), qemb)
            n = int(sc.numel())
            scores.extend(sc.detach().cpu().numpy().tolist())
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

        if include_ttd and "keystroke" in store.modalities and "keystroke" not in drop:
            for other in impostors:
                n_oks = store.n_items("keystroke", split, other)
                if n_oks <= k + k_cons:
                    continue
                seq = {m: [] for m in store.modalities}
                seq["keystroke"] = leftover[other]["keystroke"]
                ks_drop = set(store.modalities) - {"keystroke"}
                qemb = embed_views(
                    model,
                    store,
                    split,
                    other,
                    seq,
                    zstats=zstats if zscore_impostor == "claimed" else enroll_meta[other]["zstats"],
                    device=device,
                    drop=drop | ks_drop,
                    chunk=embed_chunk,
                )
                if qemb.size(0) == 0:
                    continue
                sc = similarity_score(center.unsqueeze(0).expand(qemb.size(0), -1, -1), qemb)
                ttd_seqs.append(sc.detach().cpu().numpy())
                ttd_pairs.append({"claimed": victim, "probe": other})

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or len(np.unique(y)) < 2:
        raise RuntimeError(f"No hay pares genuino/impostor en split={split}.")
    eer, tau_split = eer_and_threshold(y, s)
    auc = auc_score(y, s)
    far = far_at_frr(y, s, target_frr=0.01)
    det = det_curve(y, s)
    ttd_events = [time_to_detect(seq, tau_split, k_consecutive=k_cons) for seq in ttd_seqs]
    ttd_raw = {
        "claimed": [p["claimed"] for p in ttd_pairs],
        "probe": [p["probe"] for p in ttd_pairs],
        "events": ttd_events,
        "lengths": [int(a.size) for a in ttd_seqs],
    }

    payload = {
        "split": split,
        "n_users": len(claimed_users),
        "n_genuine": int((y == 1).sum()),
        "n_impostor": int((y == 0).sum()),
        "eer": eer,
        "auc": auc,
        "threshold_split_eer": tau_split,
        "far_at_1frr": far,
        "ttd": summarize_ttd(ttd_events),
        "det": det,
        "drop_modalities": sorted(drop),
        "k_enroll": k,
        "pair": pair,
        "collect_all": bool(collect_all),
        "zscore_impostor": zscore_impostor,
        "y_true": y.astype(int).tolist(),
        "scores": s.astype(float).tolist(),
        "probe_user_id": probe_uid,
        "claimed_user_id": claimed_uid,
        "per_user": per_user,
        "ttd_raw": ttd_raw,
        "inventory": inventory_split(store, split, k),
    }
    assert_no_raw_text(payload)
    return payload
