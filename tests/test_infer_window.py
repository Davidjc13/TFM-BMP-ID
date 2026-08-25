from bmpid.infer import score_window


def test_missing_modalities_do_not_crash():
    out = score_window({"keystroke": {"n_events": 50}, "signature": None, "scroll": None})
    assert "score" in out
    assert out["present"]["keystroke"] is True
    assert out["present"]["signature"] is False
    assert out["present"]["scroll"] is False


def test_empty_window_returns_null_score():
    out = score_window({})
    assert out["score"] == 0.0
    assert out["n_core_modalities"] == 0


def test_imu_ignored_when_not_in_proposed():
    cfg = {"imu": {"enabled_in_proposed": False}}
    out = score_window({"keystroke": {}, "imu": {"dim": 36}}, cfg)
    assert out["present"]["imu"] is False
