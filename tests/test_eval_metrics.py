import json
from pathlib import Path

import numpy as np
import pytest

from bmpid.eval.archive import persist_verify, write_det_csv
from bmpid.eval.metrics import auc_score, eer_and_threshold, far_at_frr, far_frr_at_threshold, time_to_detect
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.privacy.serialize import drop_text_fields, dump_json


def test_eer_perfect_separation():
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([1.0, 0.9, 0.8, 0.1, 0.0, -0.2])
    eer, tau = eer_and_threshold(y, s)
    assert eer == pytest.approx(0.0, abs=1e-6)
    assert auc_score(y, s) == pytest.approx(1.0, abs=1e-6)
    assert tau > 0.1


def test_far_at_1frr_and_operating_point():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.05, 0.0])
    point = far_at_frr(y, s, target_frr=0.01)
    assert "far" in point and "threshold" in point
    op = far_frr_at_threshold(y, s, 0.5)
    assert op["frr"] == pytest.approx(0.25)
    assert op["far"] == pytest.approx(0.25)


def test_ttd_consecutive():
    seq = np.array([0.9, 0.1, 0.05, 0.8])
    assert time_to_detect(seq, tau=0.2, k_consecutive=2) == 3
    assert time_to_detect(seq, tau=0.2, k_consecutive=3) is None


def test_dummy_dataset_text_not_serialized(tmp_path: Path):
    secret = "SHOULD_NOT_APPEAR_IN_JSON"
    row = {"user_id": "u1", "dt": [0.12], "ft": [0.08], "text": secret, "character": "x"}
    dest = tmp_path / "dummy.json"
    dump_json(dest, row)
    raw = dest.read_text(encoding="utf-8")
    assert secret not in raw
    loaded = json.loads(raw)
    assert "text" not in loaded
    assert "character" not in loaded
    assert loaded["user_id"] == "u1"
    assert_no_raw_text(loaded)


def test_drop_text_keeps_timings():
    cleaned = drop_text_fields({"text": "abc", "dt": [1], "nested": {"message": "no", "iki": 0.2}})
    assert "text" not in cleaned
    assert cleaned["nested"] == {"iki": 0.2}


def test_persist_verify_rejects_text_and_writes_csv(tmp_path: Path):
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.0])
    payload = {
        "split": "eval",
        "eer": 0.0,
        "auc": 1.0,
        "y_true": y.tolist(),
        "scores": s.tolist(),
        "det": {"far": [0.0, 0.0, 1.0], "frr": [1.0, 0.0, 0.0], "threshold": [1.0, 0.5, 0.0]},
        "ttd": {"n_sequences": 0},
        "text": "leak",
    }
    with pytest.raises(AssertionError):
        persist_verify(tmp_path, "eval", payload, {"seed": 1, "privacy": {"store_raw_text": False}})
    payload.pop("text")
    paths = persist_verify(tmp_path, "eval", payload, {"seed": 1, "privacy": {"store_raw_text": False}})
    csv_text = Path(paths["det_csv"]).read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "threshold,far,frr"
    assert "leak" not in csv_text
    assert "text" not in json.loads((tmp_path / "eval_summary.json").read_text(encoding="utf-8"))


def test_write_det_csv_roundtrip(tmp_path: Path):
    p = write_det_csv({"far": [0.1], "frr": [0.2], "threshold": [0.3]}, tmp_path / "d.csv")
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert lines[1] == "0.3,0.1,0.2"
