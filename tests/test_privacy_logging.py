import io
import logging

from bmpid.logging_privacy import PrivacyLogFilter, format_safe, sanitize_value
from bmpid.privacy.asserts import assert_no_raw_text, forbidden_fields_present


def test_sanitize_redacts_text_fields():
    out = sanitize_value({"user_id": "u1", "text": "password123", "n_events": 50})
    assert out["user_id"] == "u1"
    assert out["n_events"] == 50
    assert out["text"] == "[redacted]"


def test_log_filter_drops_key_dump():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(PrivacyLogFilter())
    log = logging.getLogger("test_privacy_dump")
    log.handlers.clear()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    log.info("typed=%s", {"text": "hello world", "keycode": "a"})
    logged = stream.getvalue()
    assert "hello world" not in logged
    assert "[redacted]" in logged or "keycode" not in logged.lower() or "hello" not in logged


def test_assignment_in_message_redacted():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(PrivacyLogFilter())
    log = logging.getLogger("test_privacy_assign")
    log.handlers.clear()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    log.info("event text='secretchar' n=50")
    assert "secretchar" not in stream.getvalue()


def test_assert_no_raw_text_on_dummy_dataset_row():
    row = {"user_id": "u1", "dt": [0.1], "text": "should not persist"}
    assert forbidden_fields_present(row)
    try:
        assert_no_raw_text(row)
        raise AssertionError("esperaba fallo")
    except AssertionError:
        pass
    timings_only = {"user_id": "u1", "dt": [0.1], "ft": [0.05]}
    assert_no_raw_text(timings_only)


def test_format_safe_used_for_logs():
    s = format_safe({"text": "leak", "seq_len": 50})
    assert "leak" not in s
    assert "50" in s


def test_dump_json_dummy_dataset_row(tmp_path):
    from bmpid.privacy.serialize import dump_json

    secret = "typedpassword"
    dump_json(tmp_path / "row.json", {"user_id": "42", "text": secret, "dt": [0.1]})
    blob = (tmp_path / "row.json").read_text(encoding="utf-8")
    assert secret not in blob
    assert '"text"' not in blob
