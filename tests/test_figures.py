from pathlib import Path

import pytest

from bmpid.eval.figures import generate_all, load_det_csv, plot_bars, plot_det

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_load_det_csv(tmp_path: Path):
    p = tmp_path / "det.csv"
    p.write_text("threshold,far,frr\n0,0.1,0.9\n1,0.4,0.4\n", encoding="utf-8")
    far, frr = load_det_csv(p)
    assert far == [0.1, 0.4]
    assert frr == [0.9, 0.4]


def test_plot_det_and_bars_write_png(tmp_path: Path):
    det = tmp_path / "a.csv"
    det.write_text("threshold,far,frr\n0,0.0,1.0\n1,1.0,0.0\n", encoding="utf-8")
    stem = tmp_path / "fig"
    out = plot_det([("A", det)], title="t", out_stem=stem)
    assert any(p.suffix == ".png" and p.is_file() for p in out)
    bars = plot_bars(["x", "y"], [0.2, 0.4], ylabel="EER (%)", title="b", out_stem=tmp_path / "bar")
    assert (tmp_path / "bar.png").is_file()
    assert bars


def test_generate_all_from_synthetic_csv(tmp_path: Path):
    det_dir = tmp_path / "det"
    det_dir.mkdir()
    csv_body = "threshold,far,frr\n0,0.0,1.0\n0.5,0.3,0.3\n1,1.0,0.0\n"
    for name in (
        "transformer_fused_eval_det.csv",
        "ensemble_fused_eval_meta_det.csv",
        "ensemble_fused_eval_no_scroll_signature_det.csv",
    ):
        (det_dir / name).write_text(csv_body, encoding="utf-8")
    imu_det = tmp_path / "imu_det.csv"
    imu_det.write_text(csv_body, encoding="utf-8")

    tables = {
        "transformer": [
            {
                "system": "transformer_3mod",
                "split": "eval",
                "eer": 0.25,
                "ttd_detect_rate": 0.97,
            },
            {"system": "transformer_ks_sig", "split": "eval", "eer": 0.28},
            {"system": "transformer_unimodal_keystroke", "split": "eval", "eer": 0.44},
        ],
        "ensemble": [
            {
                "system": "ensemble_meta_3mod",
                "split": "eval",
                "eer": 0.29,
                "ttd_detect_rate": 0.41,
            },
            {"system": "ensemble_unimodal_keystroke", "split": "eval", "eer": 0.47},
            {"system": "ensemble_ks_sig_fixed", "split": "eval", "eer": 0.36},
        ],
        "seqlen_keystroke": [
            {"system": "keystroke_unimodal_N25", "split": "eval", "eer": 0.49},
            {"system": "keystroke_unimodal_N50", "split": "eval", "eer": 0.47},
            {"system": "keystroke_unimodal_N100", "split": "eval", "eer": 0.46},
        ],
    }
    tables_path = tmp_path / "tables.json"
    tables_path.write_text(__import__("json").dumps(tables), encoding="utf-8")
    imu_index = tmp_path / "imu_index.json"
    imu_index.write_text(
        __import__("json").dumps(
            {
                "runs": [
                    {"tag": "eval_keystroke_unimodal", "eer": 0.47},
                    {"tag": "eval_imu_unimodal", "eer": 0.51},
                    {"tag": "eval_ks_imu_inv_eer", "eer": 0.35},
                ]
            }
        ),
        encoding="utf-8",
    )

    data_out = tmp_path / "figures"
    tex_out = tmp_path / "tex"
    manifest = generate_all(
        tables_path=tables_path,
        det_dir=det_dir,
        imu_index=imu_index,
        imu_det=imu_det,
        rf_index=tmp_path / "no_rf.json",
        data_out=data_out,
        tex_out=tex_out,
    )
    stems = [
        "det_transformer_ensemble",
        "eer_barras_propuesto",
        "eer_seqlen",
        "eer_imu",
        "ttd_detect",
    ]
    for stem in stems:
        assert (data_out / f"{stem}.png").is_file()
        assert (tex_out / f"{stem}.png").is_file()
        assert (tex_out / f"{stem}.eps").is_file()
    assert (data_out / "index.json").is_file()
    assert "eer_imu" in manifest["stems"]
    assert "eer_rf_vs_propuesto" not in manifest["stems"]


def test_generate_all_rf_figure(tmp_path: Path):
    det_dir = tmp_path / "det"
    det_dir.mkdir()
    csv_body = "threshold,far,frr\n0,0.0,1.0\n0.5,0.3,0.3\n1,1.0,0.0\n"
    for name in (
        "transformer_fused_eval_det.csv",
        "ensemble_fused_eval_meta_det.csv",
        "ensemble_fused_eval_no_scroll_signature_det.csv",
    ):
        (det_dir / name).write_text(csv_body, encoding="utf-8")
    tables = {
        "transformer": [
            {"system": "transformer_3mod", "split": "eval", "eer": 0.25, "ttd_detect_rate": 0.97},
            {"system": "transformer_ks_sig", "split": "eval", "eer": 0.28},
            {"system": "transformer_unimodal_keystroke", "split": "eval", "eer": 0.44},
        ],
        "ensemble": [
            {"system": "ensemble_meta_3mod", "split": "eval", "eer": 0.29, "ttd_detect_rate": 0.41},
            {"system": "ensemble_unimodal_keystroke", "split": "eval", "eer": 0.47},
            {"system": "ensemble_ks_sig_fixed", "split": "eval", "eer": 0.36},
        ],
        "seqlen_keystroke": [
            {"system": "keystroke_unimodal_N25", "split": "eval", "eer": 0.49},
            {"system": "keystroke_unimodal_N50", "split": "eval", "eer": 0.47},
            {"system": "keystroke_unimodal_N100", "split": "eval", "eer": 0.46},
        ],
    }
    tables_path = tmp_path / "tables.json"
    tables_path.write_text(__import__("json").dumps(tables), encoding="utf-8")
    rf_index = tmp_path / "rf.json"
    rf_index.write_text(
        __import__("json").dumps(
            {
                "runs": [
                    {"system": "rf_keystroke", "split": "eval", "eer": 0.40},
                    {"system": "rf_3mod_features", "split": "eval", "eer": 0.35},
                ]
            }
        ),
        encoding="utf-8",
    )
    data_out = tmp_path / "figures"
    tex_out = tmp_path / "tex"
    manifest = generate_all(
        tables_path=tables_path,
        det_dir=det_dir,
        imu_index=tmp_path / "missing_imu.json",
        imu_det=tmp_path / "missing_imu_det.csv",
        rf_index=rf_index,
        data_out=data_out,
        tex_out=tex_out,
    )
    assert "eer_rf_vs_propuesto" in manifest["stems"]
    assert (tex_out / "eer_rf_vs_propuesto.png").is_file()
    assert (tex_out / "eer_rf_vs_propuesto.eps").is_file()
