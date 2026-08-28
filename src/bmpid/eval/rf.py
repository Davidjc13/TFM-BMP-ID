"""Random Forest en el protocolo leftover propio (Fase 8). Sin Torch en el RF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from bmpid.config import REPO_ROOT, load_config
from bmpid.eval.archive import persist_verify, write_env
from bmpid.eval.metrics import (
    auc_score,
    det_curve,
    eer_and_threshold,
    far_at_frr,
    far_frr_at_threshold,
    summarize_ttd,
    time_to_detect,
)
from bmpid.eval.protocol import (
    _enroll_indices,
    _query_indices,
    _zstats_enroll,
    inventory_split,
    leftover_indices,
)
from bmpid.features.normalize import apply_zscore
from bmpid.logging_privacy import configure_logging, format_safe
from bmpid.models.behave_tf import CORE
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.seed import set_global_seed
from bmpid.train.dataset import NyitFeatureStore

_RF_KW = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=1,
)


def rf_archive_root(cfg: dict[str, Any]) -> Path:
    raw = Path((cfg.get("eval") or {}).get("rf_archive_root", "data/rf_run"))
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def _stats_1d(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return np.zeros(5, dtype=np.float64)
    return np.array(
        [v.mean(), v.std(), np.percentile(v, 10), np.percentile(v, 50), np.percentile(v, 90)],
        dtype=np.float64,
    )


def aggregate_keystroke(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Estadísticas de IKI (y DT=0 explícito) en la ventana N=50."""
    arr = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != m.size:
        m = np.ones(arr.shape[0], dtype=bool)
    valid = arr[m] if m.any() else arr[:0]
    dt = valid[:, 0] if valid.size else np.zeros(0)
    ft = valid[:, 1] if valid.size and valid.shape[1] > 1 else np.zeros(0)
    iki = valid[:, 2] if valid.size and valid.shape[1] > 2 else np.zeros(0)
    frac = float(m.mean()) if m.size else 0.0
    n_valid = float(m.sum())
    return np.concatenate(
        [_stats_1d(dt), _stats_1d(ft), _stats_1d(iki), np.array([frac, n_valid], dtype=np.float64)]
    )


def aggregate_signature(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Descriptores agregados de la serie T=128, C=6."""
    arr = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != m.size:
        m = np.ones(arr.shape[0], dtype=bool)
    valid = arr[m] if m.any() else np.zeros((0, arr.shape[-1]))
    parts = [_stats_1d(valid[:, c] if valid.size else np.zeros(0)) for c in range(arr.shape[-1])]
    if valid.shape[0] >= 2:
        dxy = np.diff(valid[:, : min(2, valid.shape[1])], axis=0)
        path = float(np.linalg.norm(dxy, axis=1).sum()) if dxy.size else 0.0
    else:
        path = 0.0
    duration = float(m.sum())
    return np.concatenate(parts + [np.array([path, duration], dtype=np.float64)])


def aggregate_scroll(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.reshape(-1)
    return arr.astype(np.float64)[:6] if arr.size >= 6 else np.pad(arr.reshape(-1), (0, max(0, 6 - arr.size)))


_AGG = {"keystroke": aggregate_keystroke, "signature": aggregate_signature, "scroll": aggregate_scroll}


def pair_features(template: np.ndarray, query: np.ndarray) -> np.ndarray:
    t = np.asarray(template, dtype=np.float64).reshape(-1)
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    if t.size != q.size:
        n = max(t.size, q.size)
        t = np.pad(t, (0, n - t.size))
        q = np.pad(q, (0, n - q.size))
    return np.concatenate([t, q, np.abs(t - q), t * q])


def fit_verifier_rf(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    **kwargs: Any,
) -> RandomForestClassifier:
    params = {**_RF_KW, **kwargs, "random_state": int(seed)}
    clf = RandomForestClassifier(**params)
    clf.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int))
    return clf


def rf_positive_scores(clf: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(np.asarray(X, dtype=np.float64))
    classes = list(clf.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(proba.shape[0], dtype=np.float64)


def _raw_item(store: NyitFeatureStore, modality: str, split: str, uid: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
    bank = store.banks[modality]
    ids = bank.indices(split, uid)
    if ids.size == 0:
        return store.empty_view(modality)
    j = int(ids[int(idx) % int(ids.size)])
    return bank.x[j], bank.mask[j]


def _vector_from_raw(modality: str, x: np.ndarray, mask: np.ndarray, zstats: dict[str, list[float]]) -> np.ndarray:
    xx = apply_zscore(np.asarray(x, dtype=np.float32), zstats)
    xx = np.where(np.asarray(mask)[..., None], xx, 0.0).astype(np.float32)
    return _AGG[modality](xx, mask)


def collect_feature_pairs(
    store: NyitFeatureStore,
    split: str,
    cfg: dict[str, Any],
    *,
    modalities: Sequence[str],
    pair: str = "min",
) -> dict[str, Any]:
    """Pares leftover: template = media de k enrolamientos; consulta = vector agregado.

    Las consultas se z-normalizan con el enrolamiento del **dueño** de la traza
    (una pasada por usuario) para no reaplicar z-score del reclamado en cada par.
    """
    k = int(cfg.get("k_enroll", 5))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    mods = tuple(m for m in modalities if m in store.modalities)
    users = store.users(split, require_all=False)
    templates: dict[str, dict[str, np.ndarray]] = {}
    leftover: dict[str, dict[str, list[int]]] = {}
    leftover_vec: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for uid in users:
        eidx = _enroll_indices(store, split, uid, k)
        active = [m for m in mods if len(eidx[m]) > 0]
        if not active:
            continue
        zstats = _zstats_enroll(store, split, uid, eidx)
        leftover[uid] = leftover_indices(store, split, uid, k)
        leftover_vec[uid] = {m: {} for m in mods}
        per: dict[str, np.ndarray] = {}
        for m in mods:
            if not eidx[m]:
                continue
            zs = zstats[m]
            vecs = []
            for i in eidx[m]:
                x, mask = _raw_item(store, m, split, uid, i)
                vecs.append(_vector_from_raw(m, x, mask, zs))
            per[m] = np.mean(np.stack(vecs, axis=0), axis=0)
            for i in leftover[uid][m]:
                x, mask = _raw_item(store, m, split, uid, i)
                leftover_vec[uid][m][i] = _vector_from_raw(m, x, mask, zs)
        if not per:
            continue
        templates[uid] = per

    expected_dim = {m: int(_AGG[m](np.zeros((8, 6)), np.ones(8, dtype=bool)).size) for m in mods}
    # keystroke dummy is (8,6) but aggregator uses 3 channels via shape; force real dims:
    expected_dim["keystroke"] = int(aggregate_keystroke(np.zeros((50, 3)), np.ones(50, dtype=bool)).size)
    expected_dim["signature"] = int(aggregate_signature(np.zeros((128, 6)), np.ones(128, dtype=bool)).size)
    expected_dim["scroll"] = 6
    claimed = list(templates)
    labels: list[int] = []
    feats: list[np.ndarray] = []
    present_rows: list[np.ndarray] = []
    probe_uid: list[str] = []
    claimed_uid: list[str] = []
    per_user: list[dict[str, Any]] = []
    ttd_feat_seqs: list[list[np.ndarray]] = []
    ttd_pairs: list[dict[str, str]] = []

    def _qidx(probe: str) -> dict[str, list[int]]:
        drop = set(store.modalities) - set(mods)
        return _query_indices(leftover[probe], drop, pair=pair, max_q=0, rng=rng)

    def _row(victim: str, probe: str, qidx: dict[str, list[int]], j: int) -> tuple[np.ndarray, np.ndarray] | None:
        parts: list[np.ndarray] = []
        flags = np.zeros(len(mods), dtype=np.float64)
        for i, m in enumerate(mods):
            if m not in templates[victim] or j >= len(qidx.get(m, [])):
                dim = expected_dim.get(m, 6)
                parts.append(np.zeros(dim * 4, dtype=np.float64))
                continue
            qv = leftover_vec[probe][m].get(qidx[m][j])
            if qv is None:
                parts.append(np.zeros(expected_dim.get(m, 6) * 4, dtype=np.float64))
                continue
            parts.append(pair_features(templates[victim][m], qv))
            flags[i] = 1.0
        if not flags.any():
            return None
        return np.concatenate(parts + [flags]), flags

    for victim in claimed:
        qidx_g = _qidx(victim)
        n = max((len(qidx_g.get(m, [])) for m in mods), default=0)
        n_gen = 0
        for j in range(n):
            packed = _row(victim, victim, qidx_g, j)
            if packed is None:
                continue
            vec, flags = packed
            feats.append(vec)
            present_rows.append(flags)
            labels.append(1)
            probe_uid.append(victim)
            claimed_uid.append(victim)
            n_gen += 1
        n_imp = 0
        for other in claimed:
            if other == victim:
                continue
            qidx_i = _qidx(other)
            n_i = max((len(qidx_i.get(m, [])) for m in mods), default=0)
            for j in range(n_i):
                packed = _row(victim, other, qidx_i, j)
                if packed is None:
                    continue
                vec, flags = packed
                feats.append(vec)
                present_rows.append(flags)
                labels.append(0)
                probe_uid.append(other)
                claimed_uid.append(victim)
                n_imp += 1
        per_user.append(
            {
                "user_id": victim,
                "n_genuine": n_gen,
                "n_impostor": n_imp,
                "leftover": {m: len(leftover[victim][m]) for m in store.modalities},
            }
        )
        if "keystroke" in mods and "keystroke" in templates[victim]:
            for other in claimed:
                if other == victim:
                    continue
                seq_idx = leftover[other].get("keystroke") or []
                if len(seq_idx) < 2:
                    continue
                rows = []
                for idx in seq_idx:
                    qv = leftover_vec[other]["keystroke"].get(idx)
                    if qv is None:
                        continue
                    rows.append(np.concatenate([pair_features(templates[victim]["keystroke"], qv), np.array([1.0])]))
                if not rows:
                    continue
                ttd_feat_seqs.append(rows)
                ttd_pairs.append({"claimed": victim, "probe": other})

    if not feats:
        raise RuntimeError(f"No hay pares RF en split={split} mods={mods}.")
    X = np.stack(feats, axis=0)
    y = np.asarray(labels, dtype=int)
    payload = {
        "split": split,
        "modalities": list(mods),
        "X": X,
        "y_true": y,
        "present": np.stack(present_rows, axis=0),
        "probe_user_id": probe_uid,
        "claimed_user_id": claimed_uid,
        "per_user": per_user,
        "ttd_feat_seqs": ttd_feat_seqs,
        "ttd_pairs": ttd_pairs,
        "inventory": inventory_split(store, split, k),
        "n_users": len(claimed),
        "k_enroll": k,
        "pair": pair,
    }
    assert_no_raw_text({k: v for k, v in payload.items() if k not in {"X", "y_true", "present", "ttd_feat_seqs"}})
    return payload


def _pack_metrics(
    name: str,
    collected: dict[str, Any],
    scores: np.ndarray,
    *,
    ttd_seqs: list[np.ndarray] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y = np.asarray(collected["y_true"], dtype=int)
    k_cons = 2
    eer, tau = eer_and_threshold(y, scores)
    ttd_events = [time_to_detect(seq, tau, k_consecutive=k_cons) for seq in (ttd_seqs or [])]
    extra_m = {
        "n_genuine": int((y == 1).sum()),
        "n_impostor": int((y == 0).sum()),
        "eer": eer,
        "auc": auc_score(y, scores),
        "threshold_split_eer": tau,
        "far_at_1frr": far_at_frr(y, scores, target_frr=0.01),
        "ttd": summarize_ttd(ttd_events),
        "det": det_curve(y, scores),
        "scores": np.asarray(scores, dtype=float).tolist(),
        "y_true": y.astype(int).tolist(),
    }
    ttd_raw = {
        "events": ttd_events,
        "lengths": [int(np.asarray(a).size) for a in (ttd_seqs or [])],
        "claimed": [p["claimed"] for p in collected.get("ttd_pairs", [])],
        "probe": [p["probe"] for p in collected.get("ttd_pairs", [])],
    }
    out = {
        "model": "random_forest",
        "system": name,
        "split": collected["split"],
        "modalities": collected["modalities"],
        "n_users": collected["n_users"],
        "k_enroll": collected["k_enroll"],
        "pair": collected["pair"],
        "inventory": collected.get("inventory"),
        "per_user": collected.get("per_user"),
        "probe_user_id": collected.get("probe_user_id"),
        "claimed_user_id": collected.get("claimed_user_id"),
        "ttd_raw": ttd_raw,
        **extra_m,
        **(extra or {}),
    }
    assert_no_raw_text(
        {k: v for k, v in out.items() if k not in {"det", "y_true", "scores", "probe_user_id", "claimed_user_id", "ttd_raw", "inventory", "per_user"}}
    )
    return out


def _persist(dest: Path, tag: str, payload: dict[str, Any], cfg: dict[str, Any], tau_dev: float | None) -> dict[str, Any]:
    if tau_dev is not None:
        payload = dict(payload)
        payload["operating_point_dev_tau"] = far_frr_at_threshold(
            np.asarray(payload["y_true"]),
            np.asarray(payload["scores"]),
            float(tau_dev),
        )
        payload["operating_tau_dev"] = float(tau_dev)
    persist_verify(dest, tag, payload, cfg)
    slim = {k: v for k, v in payload.items() if k not in {"det", "y_true", "scores", "probe_user_id", "claimed_user_id", "ttd_raw", "inventory", "per_user"}}
    (dest / f"{tag}_summary.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    return slim


def run_l1(cfg: dict[str, Any], store: NyitFeatureStore, dest: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("rf_keystroke", ("keystroke",), "min"),
        ("rf_3mod_features", CORE, "min"),
    ]
    seed = int(cfg.get("seed", 42))
    for name, mods, pair in specs:
        dev = collect_feature_pairs(store, "dev", cfg, modalities=mods, pair=pair)
        clf = fit_verifier_rf(dev["X"], dev["y_true"], seed=seed)
        s_dev = rf_positive_scores(clf, dev["X"])
        ttd_dev = [rf_positive_scores(clf, np.stack(seq)) for seq in dev["ttd_feat_seqs"]] if name == "rf_keystroke" else []
        m_dev = _pack_metrics(name, dev, s_dev, ttd_seqs=ttd_dev or None, extra={"level": "L1", "trained_on": "dev"})
        tau = float(m_dev["threshold_split_eer"])
        _persist(dest / name, "dev", m_dev, cfg, None)

        ev = collect_feature_pairs(store, "eval", cfg, modalities=mods, pair=pair)
        s_ev = rf_positive_scores(clf, ev["X"])
        ttd_ev = [rf_positive_scores(clf, np.stack(seq)) for seq in ev["ttd_feat_seqs"]] if name == "rf_keystroke" else []
        m_ev = _pack_metrics(name, ev, s_ev, ttd_seqs=ttd_ev or None, extra={"level": "L1", "trained_on": "dev"})
        slim = _persist(dest / name, "eval", m_ev, cfg, tau)
        rows.append(
            {
                "system": name,
                "level": "L1",
                "split": "eval",
                "eer": slim["eer"],
                "auc": slim["auc"],
                "n_genuine": slim["n_genuine"],
                "n_impostor": slim["n_impostor"],
                "n_users": slim["n_users"],
                "modalities": list(mods),
                "official": True,
            }
        )
    return rows


def run_l3(cfg: dict[str, Any], dest: Path) -> list[dict[str, Any]]:
    """RF meta sobre el triplete de scores de expertos (dev only). No reentrena encoders."""
    from bmpid.eval.ensemble import collect_modality_scores
    from bmpid.train.experts import load_expert
    from bmpid.train.transformer import _device

    seed = int(cfg.get("seed", 42))
    tcfg = (cfg.get("train") or {}).get("ensemble") or {}
    device = _device({"train": {"transformer": tcfg}})
    store = NyitFeatureStore(cfg, modalities=CORE, scroll_source="vector")
    experts = {m: load_expert(cfg, m, device) for m in CORE}
    dev = collect_modality_scores(
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
    s = np.asarray(dev["scores_modal"], dtype=np.float64)
    p = np.asarray(dev["present"], dtype=np.float64)
    X_dev = np.concatenate([s, p], axis=1)
    y_dev = np.asarray(dev["y_true"], dtype=int)
    clf = fit_verifier_rf(X_dev, y_dev, seed=seed)
    s_dev = rf_positive_scores(clf, X_dev)
    fake_dev = {**dev, "modalities": list(dev["modalities"]), "ttd_pairs": [], "ttd_feat_seqs": []}
    m_dev = _pack_metrics("rf_meta_3mod", fake_dev, s_dev, extra={"level": "L3", "trained_on": "dev"})
    tau = float(m_dev["threshold_split_eer"])
    _persist(dest / "rf_meta_3mod", "dev", m_dev, cfg, None)

    ev = collect_modality_scores(
        experts,
        store,
        "eval",
        cfg,
        device=device,
        collect_all=True,
        pair="min",
        zscore_impostor="claimed",
        include_ttd=True,
    )
    X_ev = np.concatenate([np.asarray(ev["scores_modal"], dtype=np.float64), np.asarray(ev["present"], dtype=np.float64)], axis=1)
    s_ev = rf_positive_scores(clf, X_ev)
    mods = list(ev["modalities"])
    ttd_seqs: list[np.ndarray] = []
    if "keystroke" in mods:
        j = mods.index("keystroke")
        for seq in ev.get("ttd_seqs") or []:
            arr = np.asarray(seq, dtype=np.float64).reshape(-1)
            sm = np.zeros((arr.size, len(mods)))
            pm = np.zeros((arr.size, len(mods)))
            sm[:, j] = arr
            pm[:, j] = 1.0
            ttd_seqs.append(rf_positive_scores(clf, np.concatenate([sm, pm], axis=1)))
    fake_ev = {**ev, "ttd_feat_seqs": [], "ttd_pairs": ev.get("ttd_pairs", [])}
    m_ev = _pack_metrics("rf_meta_3mod", fake_ev, s_ev, ttd_seqs=ttd_seqs or None, extra={"level": "L3", "trained_on": "dev"})
    slim = _persist(dest / "rf_meta_3mod", "eval", m_ev, cfg, tau)
    return [
        {
            "system": "rf_meta_3mod",
            "level": "L3",
            "split": "eval",
            "eer": slim["eer"],
            "auc": slim["auc"],
            "n_genuine": slim["n_genuine"],
            "n_impostor": slim["n_impostor"],
            "n_users": slim["n_users"],
            "modalities": list(CORE),
            "compared_to": "ensemble_meta_3mod",
            "official": True,
        }
    ]


def run_rf(cfg: dict[str, Any] | None = None, *, skip_l3: bool = False) -> dict[str, Any]:
    log = configure_logging()
    cfg = cfg or load_config()
    set_global_seed(int(cfg["seed"]))
    dest = rf_archive_root(cfg)
    env = write_env(cfg)
    (dest / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    store = NyitFeatureStore(cfg, modalities=CORE, scroll_source="vector")
    rows = run_l1(cfg, store, dest)
    if not skip_l3:
        rows.extend(run_l3(cfg, dest))
    index = {
        "env": env,
        "protocol": {
            "dataset": "nyit",
            "splits": "user-disjoint train/dev/eval",
            "k_enroll": int(cfg.get("k_enroll", 5)),
            "seq_len": int(cfg.get("seq_len", 50)),
            "impostor": "random leftover of other users",
            "note": "RF no replica Cariello (otro split, mocap Vicon, ventanas 1 s).",
            "store_raw_text": False,
        },
        "runs": rows,
    }
    assert_no_raw_text(index["protocol"])
    (dest / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    log.info("rf %s", format_safe({"n_runs": len(rows), "out": str(dest)}))
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval Random Forest leftover NYIT (L1 rasgos, L3 meta).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-l3", action="store_true", help="Solo L1 (sin reconsultar expertos).")
    args = parser.parse_args(argv)
    print(json.dumps(run_rf(load_config(args.config), skip_l3=bool(args.skip_l3)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
