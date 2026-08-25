from bmpid.config import load_config


def test_default_config_fase0_keys():
    cfg = load_config()
    assert cfg["seq_len"] == 50
    assert cfg["seq_len_keystroke"] == 50
    assert cfg["set2set_margin"] == 0.2
    assert cfg["triplet_margin"] == 0.2
    assert cfg["k_enroll"] == 5
    assert cfg["privacy"]["store_raw_text"] is False
    assert cfg["imu"]["enabled_in_proposed"] is False
    assert cfg["dataset"]["name"] == "nyit"
    assert cfg["signature"]["decision"] == "proxy_touch"
    assert cfg["scroll"]["task"] == "nyit_swipe"
    assert cfg["splits"]["disjoint_users"] is True
    assert cfg["ttd"]["rule"] in {"first_cross", "consecutive"}


def test_store_raw_text_cannot_be_true(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("seed: 1\nprivacy:\n  store_raw_text: true\n", encoding="utf-8")
    try:
        load_config(p)
        raise AssertionError("debía rechazar store_raw_text")
    except ValueError as exc:
        assert "store_raw_text" in str(exc)
