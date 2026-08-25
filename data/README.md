# Datos — NYIT Smartphone Activity Dataset

Corpus **activo en Fase 1**. No se versiona el dump crudo (demasiado grande). Checksums de tecleo y swipe: `data/checksums/nyit_core.sha256`.

## Origen

- URL: <https://www.nyit-lamp.com/dataset/dataset4/>
- Cita: `NYIT_dataset:2025` (README del dump)
- Copia local: `nyit_smartphone_dataset_home_lab_cleaned/` (home + lab, cleaned)
- Licencia / ética: recogida con aprobación IRB NYIT; datos biométricos conductuales (minimización: el manifiesto **no** guarda caracteres).

## Qué se usa

| Modalidad núcleo | Ficheros | Notas |
| --- | --- | --- |
| Tecleo | `keystrokes_data.csv` | Solo timestamps en el pipeline; el carácter no se persiste. Solo hay eventos de pulsación (no `keyup`) → DT/FT se derivarán en Fase 2 con esa limitación. |
| Scroll | `swipe_data.csv` | Trayectorias `(t, x, y)` en `sit_and_swipe` / `walk_and_swipe` / `home/swiping`. Analogía gallery swiping. |
| Firma | **proxy** = mismas trayectorias de swipe | No es firma manuscrita legal. `signature.decision: proxy_touch`. Tecleo y swipe **no** coinciden en el mismo trial. |
| IMU | `accelerometer.csv`, `gyroscope.csv` | En manifiesto; **fuera** del sistema propuesto. |

## Ingesta

```bash
python -m bmpid.data --config configs/default.yaml
python -m bmpid.data --verify-checksums
```

Salida: `data/processed/nyit/manifest.parquet`, `data/processed/nyit/splits.json`.

## Features (Fase 2)

```bash
python -m bmpid.features --config configs/default.yaml
```

Tensores en `data/processed/nyit/features/`. Qué se hizo y por qué: [`PREPROCESAMIENTO.md`](PREPROCESAMIENTO.md).

Splits **usuario-disjuntos** (`train` / `dev` / `eval`, ratios 0.6 / 0.2 / 0.2, `seed: 42`). Un `user_id` de lab y home permanece en un único split.
