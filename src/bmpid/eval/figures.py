"""Figuras Fase 7 a partir de DET CSV y tables.json (sin reentrenar)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from bmpid.config import REPO_ROOT

DET_DIR = REPO_ROOT / "data" / "eval_run" / "det"
TABLES = REPO_ROOT / "data" / "eval_run" / "tables.json"
IMU_INDEX = REPO_ROOT / "data" / "imu_run" / "index.json"
IMU_DET = REPO_ROOT / "data" / "imu_run" / "fused" / "eval_ks_imu_inv_eer_det.csv"
RF_INDEX = REPO_ROOT / "data" / "rf_run" / "index.json"

# Paleta acotada (colorblind-friendly).
_C = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9")


def _thesis_fig_dir() -> Path:
    return (
        REPO_ROOT
        / "project_tex"
        / "Plantilla_TFM_VIU__Universidad_Internacional_Valencia_"
        / "files"
        / "figures"
        / "cap06"
    )


def load_det_csv(path: Path) -> tuple[list[float], list[float]]:
    far: list[float] = []
    frr: list[float] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f = float(row["far"])
            r = float(row["frr"])
            if f != f or r != r:  # NaN
                continue
            far.append(f)
            frr.append(r)
    return far, frr


def _save(fig: Any, dests: Sequence[Path]) -> None:
    import matplotlib.pyplot as plt

    for dest in dests:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_det(
    series: list[tuple[str, Path]],
    *,
    title: str,
    out_stem: Path,
    footnote: str | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    plotted = 0
    for i, (label, path) in enumerate(series):
        if not path.is_file():
            continue
        far, frr = load_det_csv(path)
        ax.plot(far, frr, color=_C[i % len(_C)], label=label, lw=1.8)
        plotted += 1
    ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=0.8, label="Azar")
    ax.set_xlabel("Tasa de falsa aceptación (FAR)")
    ax.set_ylabel("Tasa de falso rechazo (FRR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="0.85")
    if footnote:
        fig.text(0.02, 0.01, footnote, fontsize=7, color="0.3")
        fig.subplots_adjust(bottom=0.18)
    else:
        fig.tight_layout()
    if plotted == 0:
        plt.close(fig)
        return []
    paths = [out_stem.with_suffix(".png"), out_stem.with_suffix(".eps")]
    _save(fig, paths)
    return paths


def plot_bars(
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    out_stem: Path,
    as_percent: bool = True,
    colors: Sequence[str] | None = None,
    footnote: str | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(labels))
    y = [v * 100.0 if as_percent else v for v in values]
    cols = list(colors) if colors else [_C[i % len(_C)] for i in range(len(labels))]
    ax.bar(x, y, color=cols, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", color="0.85")
    if footnote:
        fig.text(0.02, 0.01, footnote, fontsize=7, color="0.3")
        fig.subplots_adjust(bottom=0.28)
    else:
        fig.tight_layout()
    paths = [out_stem.with_suffix(".png"), out_stem.with_suffix(".eps")]
    _save(fig, paths)
    return paths


def _eer(tables: dict[str, Any], section: str, system: str, split: str = "eval") -> float:
    for row in tables[section]:
        if row.get("system") == system and row.get("split") == split:
            return float(row["eer"])
    raise KeyError(system)


def _copy_stems(src: Path, dest: Path) -> list[str]:
    written: list[str] = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    for suf in (".png", ".eps"):
        s = src.with_suffix(suf)
        if s.is_file():
            d = dest.with_suffix(suf)
            if s.resolve() != d.resolve():
                shutil.copyfile(s, d)
            written.append(str(d))
    return written


def generate_all(
    cfg: dict[str, Any] | None = None,
    *,
    tables_path: Path | None = None,
    det_dir: Path | None = None,
    imu_index: Path | None = None,
    imu_det: Path | None = None,
    rf_index: Path | None = None,
    data_out: Path | None = None,
    tex_out: Path | None = None,
) -> dict[str, Any]:
    del cfg
    tables_path = Path(tables_path) if tables_path else TABLES
    det_dir = Path(det_dir) if det_dir else DET_DIR
    imu_index = Path(imu_index) if imu_index else IMU_INDEX
    imu_det = Path(imu_det) if imu_det else IMU_DET
    rf_index = Path(rf_index) if rf_index else RF_INDEX
    data_out = Path(data_out) if data_out else REPO_ROOT / "data" / "figures"
    tex_out = Path(tex_out) if tex_out else _thesis_fig_dir()
    data_out.mkdir(parents=True, exist_ok=True)
    tex_out.mkdir(parents=True, exist_ok=True)
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    written: list[str] = []
    note_n = "N genuinos: 545 (Transformer 3-mod) frente a 2278 (ensemble y ablaciones de tecleo)."

    def dual(name: str) -> tuple[Path, Path]:
        return data_out / name, tex_out / name

    d1, t1 = dual("det_transformer_ensemble")
    series = [
        ("Transformer 3-mod", det_dir / "transformer_fused_eval_det.csv"),
        ("Ensemble meta", det_dir / "ensemble_fused_eval_meta_det.csv"),
        ("Ensemble tecleo", det_dir / "ensemble_fused_eval_no_scroll_signature_det.csv"),
    ]
    if imu_det.is_file():
        series.append(("Tecleo+IMU (réplica, no propuesto)", imu_det))
    plot_det(
        series,
        title="DET — eval leftover (NYIT)",
        out_stem=d1,
        footnote=note_n,
    )
    written.extend(_copy_stems(d1, t1))

    d2, t2 = dual("eer_barras_propuesto")
    labels = [
        "TF uni. tecleo",
        "TF ks+firma",
        "TF 3-mod",
        "Ens. uni. tecleo",
        "Ens. ks+firma",
        "Ens. meta 3-mod",
    ]
    eers = [
        _eer(tables, "transformer", "transformer_unimodal_keystroke"),
        _eer(tables, "transformer", "transformer_ks_sig"),
        _eer(tables, "transformer", "transformer_3mod"),
        _eer(tables, "ensemble", "ensemble_unimodal_keystroke"),
        _eer(tables, "ensemble", "ensemble_ks_sig_fixed"),
        _eer(tables, "ensemble", "ensemble_meta_3mod"),
    ]
    plot_bars(
        labels,
        eers,
        ylabel="EER (%)",
        title="EER eval — sistema propuesto (NYIT)",
        out_stem=d2,
        footnote=note_n + " IMU no figura aquí.",
    )
    written.extend(_copy_stems(d2, t2))

    d3, t3 = dual("eer_seqlen")
    nlab = ["N=25", "N=50", "N=100"]
    nval = [
        _eer(tables, "seqlen_keystroke", "keystroke_unimodal_N25"),
        _eer(tables, "seqlen_keystroke", "keystroke_unimodal_N50"),
        _eer(tables, "seqlen_keystroke", "keystroke_unimodal_N100"),
    ]
    plot_bars(nlab, nval, ylabel="EER (%)", title="Tecleo BiLSTM — longitud de ventana (NYIT)", out_stem=d3)
    written.extend(_copy_stems(d3, t3))

    imu = json.loads(imu_index.read_text(encoding="utf-8")) if imu_index.is_file() else None
    if imu:
        by_tag = {r["tag"]: r["eer"] for r in imu["runs"] if "eer" in r}
        d4, t4 = dual("eer_imu")
        plot_bars(
            ["Tecleo", "IMU", "Tecleo+IMU (1/EER)"],
            [
                by_tag["eval_keystroke_unimodal"],
                by_tag["eval_imu_unimodal"],
                by_tag["eval_ks_imu_inv_eer"],
            ],
            ylabel="EER (%)",
            title="Réplica IMU (no propuesto) — eval NYIT",
            out_stem=d4,
            footnote="No comparable 1:1 con BehaveFormer en HuMIdb (EER 2,95 %).",
        )
        written.extend(_copy_stems(d4, t4))

    d5, t5 = dual("ttd_detect")
    tf3 = next(r for r in tables["transformer"] if r["system"] == "transformer_3mod" and r["split"] == "eval")
    em = next(r for r in tables["ensemble"] if r["system"] == "ensemble_meta_3mod" and r["split"] == "eval")
    plot_bars(
        ["Transformer 3-mod", "Ensemble meta"],
        [float(tf3["ttd_detect_rate"]), float(em["ttd_detect_rate"])],
        ylabel="Tasa de detección TTD (%)",
        title="TTD (k=2 ventanas consecutivas) — eval NYIT",
        out_stem=d5,
        as_percent=True,
    )
    written.extend(_copy_stems(d5, t5))

    stems = [
        "det_transformer_ensemble",
        "eer_barras_propuesto",
        "eer_seqlen",
        "eer_imu",
        "ttd_detect",
    ]
    rf = json.loads(rf_index.read_text(encoding="utf-8")) if rf_index.is_file() else None
    if rf:
        by_sys = {r["system"]: r["eer"] for r in rf.get("runs", []) if r.get("split") == "eval" and "eer" in r}
        d7, t7 = dual("eer_rf_vs_propuesto")
        plot_bars(
            ["RF tecleo", "RF 3-mod (rasgos)", "Ensemble meta", "Transformer 3-mod"],
            [
                float(by_sys["rf_keystroke"]),
                float(by_sys["rf_3mod_features"]),
                _eer(tables, "ensemble", "ensemble_meta_3mod"),
                _eer(tables, "transformer", "transformer_3mod"),
            ],
            ylabel="EER (%)",
            title="EER eval — RF vs sistema propuesto (mismo leftover NYIT)",
            out_stem=d7,
            footnote=note_n + " Cariello 1,5 % no está en el eje (otro protocolo, mocap).",
        )
        written.extend(_copy_stems(d7, t7))
        stems.append("eer_rf_vs_propuesto")

    manifest = {
        "out_data": str(data_out),
        "out_tex": str(tex_out),
        "files": written,
        "stems": stems,
        "note": "EER en barras = tanto por uno × 100. IMU no es sistema propuesto. F7 no pinta Cariello 1,5 %.",
    }
    (data_out / "index.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera figuras Fase 7 (DET, EER, TTD).")
    parser.add_argument("--config", default=None, help="Reservado; no se reentrena.")
    args = parser.parse_args(argv)
    del args
    print(json.dumps(generate_all(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
