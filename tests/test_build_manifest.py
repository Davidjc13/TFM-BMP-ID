from pathlib import Path

import yaml

from bmpid.config import load_config
from bmpid.data.build import build
from bmpid.data.manifest import read_manifest
from bmpid.data.splits import assert_disjoint_user_splits
from tests.test_data_nyit import _mini_nyit


def test_build_writes_manifest_and_splits(tmp_path: Path):
    root = tmp_path / "nyit"
    _mini_nyit(root)
    cfg_path = tmp_path / "cfg.yaml"
    cfg = load_config()
    cfg["dataset"]["root"] = str(root)
    cfg["paths"]["processed_root"] = str(tmp_path / "processed")
    cfg["dataset"]["checksums_file"] = str(tmp_path / "nyit.sha256")
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loaded = load_config(cfg_path)
    summary = build(loaded, write_checksums=True)
    assert summary["n_users"] == 5
    assert summary["n_eval"] >= 1
    man = Path(summary["manifest"])
    rows = read_manifest(man)
    user_to_split = {r["user_id"]: r["split"] for r in rows}
    assert_disjoint_user_splits(user_to_split)
    text = man.read_bytes()
    assert b"SECRETCHAR" not in text
    sums = Path(summary["checksums"])
    assert sums.is_file()
    assert sums.stat().st_size > 0
