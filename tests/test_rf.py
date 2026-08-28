from pathlib import Path

import numpy as np
import pytest

from bmpid.eval.metrics import auc_score, eer_and_threshold
from bmpid.eval.rf import (
    aggregate_keystroke,
    aggregate_scroll,
    aggregate_signature,
    fit_verifier_rf,
    pair_features,
    rf_positive_scores,
)
from bmpid.privacy.asserts import assert_no_raw_text


def test_aggregates_shapes_and_explicit_dt_zero():
    x = np.zeros((50, 3), dtype=np.float32)
    x[:, 1] = 0.12
    x[:, 2] = 0.12
    mask = np.ones(50, dtype=bool)
    ks = aggregate_keystroke(x, mask)
    assert ks.shape[0] == 17
    assert ks[0] == pytest.approx(0.0)
    sig = aggregate_signature(np.random.default_rng(0).normal(size=(128, 6)), np.ones(128, dtype=bool))
    assert sig.shape[0] == 6 * 5 + 2
    sc = aggregate_scroll(np.arange(6, dtype=np.float32), np.ones(1, dtype=bool))
    assert sc.shape == (6,)
    assert sc[5] == pytest.approx(5.0)


def test_rf_eer_separable_vs_chance(tmp_path: Path):
    rng = np.random.default_rng(42)
    n_g, n_i, d = 80, 80, 8
    t_g = rng.normal(1.0, 0.05, size=(n_g, d))
    q_g = rng.normal(1.0, 0.05, size=(n_g, d))
    t_i = rng.normal(1.0, 0.05, size=(n_i, d))
    q_i = rng.normal(-1.0, 0.05, size=(n_i, d))
    X = np.stack([pair_features(t, q) for t, q in zip(np.vstack([t_g, t_i]), np.vstack([q_g, q_i]))])
    y = np.array([1] * n_g + [0] * n_i)
    clf = fit_verifier_rf(X, y, seed=42)
    s = rf_positive_scores(clf, X)
    eer, _ = eer_and_threshold(y, s)
    assert eer < 0.05
    assert auc_score(y, s) > 0.95
    out = tmp_path / "sep.json"
    payload = {"eer": float(eer), "n_genuine": int(n_g), "n_impostor": int(n_i), "user_id": "0034"}
    assert_no_raw_text(payload)
    out.write_text(__import__("json").dumps(payload), encoding="utf-8")
    assert "text" not in out.read_text(encoding="utf-8")

    Xr = rng.normal(size=(400, X.shape[1]))
    yr = rng.integers(0, 2, size=400)
    clf_r = fit_verifier_rf(Xr[:200], yr[:200], seed=42)
    sr = rf_positive_scores(clf_r, Xr[200:])
    eer_r, _ = eer_and_threshold(yr[200:], sr)
    assert eer_r > 0.15


def test_rf_does_not_need_nyit_dump():
    import bmpid.eval.rf as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "motion_capture_data" not in src
    assert "parse_nyit" not in src
