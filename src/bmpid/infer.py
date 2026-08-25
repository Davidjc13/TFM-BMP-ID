"""Inferencia de una ventana multimodal → score de similitud (mayor = más genuino).

Stub de Fase 0: no carga pesos; comprueba que modalidades faltantes no crashean
y que no se registra texto crudo.
"""

from __future__ import annotations

import argparse
from typing import Any

from bmpid.config import load_config
from bmpid.logging_privacy import configure_logging, format_safe
from bmpid.privacy.asserts import assert_no_raw_text
from bmpid.seed import set_global_seed

CORE = ("keystroke", "signature", "scroll")


def score_window(
    window: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Acepta una ventana con 0–3 modalidades núcleo; IMU se ignora por defecto."""
    cfg = cfg or {}
    present = {m: window.get(m) is not None for m in CORE}
    imu_ok = bool((cfg.get("imu") or {}).get("enabled_in_proposed", False))
    if window.get("imu") is not None and not imu_ok:
        present_imu = False
    else:
        present_imu = window.get("imu") is not None and imu_ok

    n_core = sum(present.values())
    # Placeholder: score nulo documentado si no hay ninguna modalidad.
    score = 0.0 if n_core == 0 else float(n_core) / float(len(CORE))
    return {
        "score": score,
        "present": {**present, "imu": present_imu},
        "n_core_modalities": n_core,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score de una ventana multimodal (stub Fase 0).")
    parser.add_argument("--config", default=None, help="Ruta a YAML")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(int(cfg["seed"]))
    log = configure_logging()

    # Ventana de ejemplo: solo tecleo (firma y scroll ausentes).
    demo = {
        "user_id": "u_demo",
        "session_id": "s1",
        "keystroke": {"n_events": int(cfg["seq_len"]), "features": "dt_ft_iki"},
        "signature": None,
        "scroll": None,
    }
    assert_no_raw_text(demo)
    out = score_window(demo, cfg)
    log.info("inferencia %s", format_safe({"n_events": demo["keystroke"]["n_events"], **out}))
    print(out["score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
