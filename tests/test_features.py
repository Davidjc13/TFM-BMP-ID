import ast
from pathlib import Path

import numpy as np

from bmpid.config import load_config
from bmpid.features.extract import extract
from bmpid.features.keystroke import timings_from_presses, windows_from_csv
from bmpid.features.normalize import apply_clip, fit_clip
from bmpid.features.scroll import scroll_gesture_to_vector
from bmpid.features.signature import resample_trace
from bmpid.features.windowing import sliding_windows
from bmpid.privacy.asserts import assert_no_raw_text
from tests.test_data_nyit import _mini_nyit, _write_ks


def test_keystroke_press_only_dt_zero_ft_eq_iki():
    t = np.array([0.0, 100.0, 250.0])
    x = timings_from_presses(t)
    assert x.shape == (2, 3)
    assert np.allclose(x[:, 0], 0.0)
    assert np.allclose(x[:, 1], x[:, 2])
    assert np.allclose(x[0, 2], 0.1)
    assert np.allclose(x[1, 2], 0.15)


def test_keystroke_windows_pad_and_no_chars_on_disk(tmp_path: Path):
    secret = "UNIQUECHARXYZ"
    p = tmp_path / "keystrokes_data.csv"
    _write_ks(p, [(1_600_000_000_000 + j * 80, secret) for j in range(8)])
    w, m = windows_from_csv(p, n=50, stride=25)
    assert w.shape == (1, 50, 3)
    assert int(m.sum()) == 7
    dest = tmp_path / "ks.npz"
    np.savez_compressed(dest, x=w, mask=m)
    blob = dest.read_bytes()
    assert secret.encode() not in blob
    assert secret.lower().encode() not in blob


def test_windowing_sliding():
    seq = np.arange(100, dtype=np.float32).reshape(100, 1)
    w, m = sliding_windows(seq, n=50, stride=25)
    assert w.shape[1:] == (50, 1)
    assert w.shape[0] >= 3
    assert m.all()


def test_signature_resample_fixed_T():
    t = np.linspace(0, 500, 40)
    g = np.column_stack([t, np.linspace(0, 10, 40), np.linspace(0, 20, 40)])
    feat, mask = resample_trace(g, fs=100, t_len=128)
    assert feat.shape == (128, 6)
    assert mask.shape == (128,)
    assert int(mask.sum()) >= 2


def test_scroll_vertical_line():
    t = np.arange(10, dtype=np.float64) * 100.0
    x = np.zeros(10)
    y = np.arange(10, dtype=np.float64) * 10.0
    g = np.column_stack([t, x, y])
    v = scroll_gesture_to_vector(g)
    assert v.shape == (6,)
    assert abs(v[3] - 90.0) < 1e-3
    assert abs(v[4] - 0.9) < 1e-5
    assert v[5] < 1e-5
    assert abs(v[0] - 100.0) < 1.0
    assert abs(v[1] - 100.0) < 1.0


def test_scroll_two_points_does_not_crash():
    g = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 10.0]])
    v = scroll_gesture_to_vector(g)
    assert v.shape == (6,)
    assert v[3] > 0


def test_clip_fitted_on_train_not_eval():
    train = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float32)
    eval_ = np.array([[50.0, 50.0, 50.0]], dtype=np.float32)
    bounds = fit_clip(train, low=1, high=99)
    clipped = apply_clip(eval_, bounds)
    assert float(clipped.max()) < 1.0
    assert float(clipped.max()) < 50.0


def test_extract_mini_nyit_no_text(tmp_path: Path):
    root = tmp_path / "nyit"
    _mini_nyit(root)
    cfg = load_config()
    cfg["dataset"]["root"] = str(root)
    cfg["paths"]["processed_root"] = str(tmp_path / "processed")
    cfg["dataset"]["checksums_file"] = str(tmp_path / "nyit.sha256")
    summary = extract(cfg, include_imu=False)
    assert "keystroke" in summary["counts"]
    assert "scroll" in summary["counts"]
    assert "signature" in summary["counts"]
    assert summary["imu_written"] is False
    ks = Path(summary["files"]["keystroke"])
    assert b"SECRETCHAR" not in ks.read_bytes()
    stats = Path(summary["norm_stats"]).read_text(encoding="utf-8")
    assert "SECRETCHAR" not in stats
    assert_no_raw_text(__import__("json").loads(stats))


def test_features_package_does_not_export_imu():
    import bmpid.features as feat

    assert "imu" not in feat.__all__
    src = Path(feat.__file__).read_text(encoding="utf-8")
    assert "features.imu" not in src


def test_train_module_does_not_import_imu():
    import ast

    import bmpid.train as train

    tree = ast.parse(Path(train.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("features.imu"):
            raise AssertionError("train no debe importar IMU")
        if isinstance(node, ast.Import):
            for n in node.names:
                if "features.imu" in n.name:
                    raise AssertionError("train no debe importar IMU")


def test_extract_does_not_import_imu_at_module_level():
    src = Path(__file__).resolve().parents[1] / "src" / "bmpid" / "features" / "extract.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "bmpid.features.imu":
            top_imports.append(node)
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name == "bmpid.features.imu":
                    top_imports.append(node)
    assert top_imports == []


def test_imu_exists_and_dim_36():
    from bmpid.features.imu import IMU_DIM, imu_timestep_features

    acc = np.column_stack(
        [np.arange(32, dtype=np.float64), np.ones(32), np.zeros(32), np.zeros(32)]
    )
    feat = imu_timestep_features(acc, None)
    assert IMU_DIM == 36
    assert feat.shape == (32, 36)
