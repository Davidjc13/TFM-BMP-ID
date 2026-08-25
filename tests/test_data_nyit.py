from pathlib import Path

from bmpid.config import load_config
from bmpid.data.checksums import sha256_file, verify_sha256sums, write_sha256sums
from bmpid.data.manifest import read_manifest, write_manifest
from bmpid.data.nyit import discover_users, iter_manifest_rows
from bmpid.data.splits import assert_disjoint_user_splits, assign_user_splits
from bmpid.privacy.asserts import assert_no_raw_text, forbidden_fields_present


def _write_ks(path: Path, pairs: list[tuple[float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Keystrokes Data", " Epoch Time, Keystrokes "]
    for ts, ch in pairs:
        lines.append(f"{ts},{ch}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_swipe(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Swipes Data", " Epoch Time, X-Coordinate, Y-Coordinate"]
    for ts, x, y in points:
        lines.append(f"{ts},{x},{y}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mini_nyit(root: Path) -> None:
    secret = "SECRETCHAR"
    for i, uid in enumerate(["1001", "1002", "1003", "1004", "1005"]):
        tdir = root / "lab" / uid / "session_1" / "sit_and_type" / f"01-01-2020_10-0{i}"
        _write_ks(
            tdir / "keystrokes_data.csv",
            [(1_600_000_000_000 + j * 80, secret if j % 2 == 0 else "z") for j in range(8)],
        )
        (tdir / "accelerometer.csv").write_text(
            "Accelerometer Data\n Epoch Time,X,Y,Z\n1,0,0,1\n2,0,0,1\n",
            encoding="utf-8",
        )
        sdir = root / "lab" / uid / "session_1" / "sit_and_swipe" / f"01-01-2020_11-0{i}"
        pts = [(1000.0 + k * 10, 10.0, 20.0 + k) for k in range(5)]
        pts += [(1000.0 + 800 + k * 10, 11.0, 21.0) for k in range(3)]
        _write_swipe(sdir / "swipe_data.csv", pts)


def test_manifest_has_no_raw_text(tmp_path: Path):
    _mini_nyit(tmp_path)
    users = discover_users(tmp_path)
    mapping = assign_user_splits(users, seed=42)
    rows = list(iter_manifest_rows(tmp_path, mapping, include_imu=True))
    assert rows
    blob = " ".join(str(r) for r in rows)
    assert "SECRETCHAR" not in blob
    for rec in rows:
        assert_no_raw_text(rec)
        assert not forbidden_fields_present(rec)
    mods = {r["modality"] for r in rows}
    assert {"keystroke", "scroll", "signature", "imu_acc"} <= mods
    dest = tmp_path / "manifest.jsonl"
    write_manifest(rows, dest)
    loaded = read_manifest(dest)
    text = dest.read_text(encoding="utf-8")
    assert "SECRETCHAR" not in text
    assert all("user_id" in r and "split" in r for r in loaded)


def test_corrupt_keystroke_ok_false(tmp_path: Path):
    tdir = tmp_path / "lab" / "1111" / "session_1" / "sit_and_type" / "01-01-2020_00-00"
    tdir.mkdir(parents=True)
    (tdir / "keystrokes_data.csv").write_bytes(b"\x00" * 64)
    mapping = {"1111": "train"}
    rows = list(iter_manifest_rows(tmp_path, mapping, include_imu=False))
    ks = [r for r in rows if r["modality"] == "keystroke"]
    assert len(ks) == 1
    assert ks[0]["n_events"] == 0
    assert ks[0]["ok"] is False


def test_checksum_roundtrip(tmp_path: Path):
    f = tmp_path / "a.csv"
    f.write_text("1,0,0\n", encoding="utf-8")
    sums = tmp_path / "SHA256SUMS"
    write_sha256sums(tmp_path, ["a.csv"], sums)
    assert sha256_file(f) in sums.read_text(encoding="utf-8")
    assert verify_sha256sums(tmp_path, sums) == []
    f.write_text("changed\n", encoding="utf-8")
    assert verify_sha256sums(tmp_path, sums) == ["a.csv"]


def test_default_config_nyit_signature_decision():
    cfg = load_config()
    assert cfg["dataset"]["name"] == "nyit"
    assert cfg["signature"]["decision"] == "proxy_touch"
    assert cfg["signature"]["same_trial_as_keystroke"] is False
    assert cfg["signature"]["pressure_available"] is False
    assert cfg["scroll"]["task"] == "nyit_swipe"
    assert cfg["splits"]["disjoint_users"] is True
    assert_disjoint_user_splits({"a": "train", "b": "eval"})
