# Random Forest en el leftover propio (Fase 8)

Registro de la comparación **Transformer / ensemble vs verificador clásico tipo RF** sobre NYIT, **sin** clonar el protocolo de Cariello et al. Diseño: `project/10-comparacion-rf.md`. Cifras oficiales a copiar a LaTeX: `data/rf_run/index.json`. Figura F7: `data/figures/eer_rf_vs_propuesto.png`.

Reproducir:

```bash
python -m bmpid.eval.rf --config configs/default.yaml
python -m bmpid.eval.figures --config configs/default.yaml
pytest tests/test_rf.py
```

Sklearn `RandomForestClassifier` (semilla 42). No se reentrena el Transformer ni los expertos. No se parsea mocap (`motion_capture_data__*.csv`). No se persistió texto.

## Qué se iguala y qué no

| Igualado | No igualado (otro experimento) |
| --- | --- |
| Corpus NYIT home+lab cleaned | Subset solo lab + Vicon |
| Splits user-disjoint 49/16/18, seed 42 | Train=sesión 1 / test=sesión 2 del mismo usuario |
| Leftover, k=5, impostor aleatorio | Ventanas de 1 s ancladas al evento |
| N=50 official (tecleo press-only, DT=0) | Ingeniería de rasgos + Plug-in Gait de Cariello |
| Métricas EER / AUC / FAR@1% FRR | Pintar 1,5 % y 24,6 % en el mismo eje sin nota |

Pregunta: *en este leftover press-only, ¿un RF de rasgos (o de scores) bate al Transformer 3-mod o al ensemble meta?*

**Respuesta:** no. El mejor RF aquí es el meta de scores (EER eval **0.303**). El Transformer 3-mod sigue en **0.246** y el ensemble meta en **0.290**. El RF de tecleo (**0.320**) sí mejora el unimodal BiLSTM de tecleo (**0.476**).

Eso **no** es el 1,5 % de Cariello: otro split, mocap de laboratorio, ventanas de 1 s.

## Protocolo

- Enrolamiento: k=5 primeras ventanas por usuario del split.
- Consultas: leftover completo; impostor = leftover de otros usuarios del mismo split.
- RF L1 y L3 se **ajustan solo en *dev***; el EER comparable es el de *eval* leftover.
- Umbral operativo: τ de EER en *dev*, aplicado en eval (`operating_point_dev_tau`). El EER de la fila eval usa el umbral de EER **del propio split** (igual que el resto de tablas del TFM).
- L1 3-mod y L3 usan 2278 genuinos / 38726 impostores / 18 usuarios (mismo N que el ensemble 3-mod). El Transformer 3-mod sigue con 545 genuinos (tope scroll): no mezclar N al leer F7.

## Qué se entrenó

**L1 — RF sobre rasgos** (sin Vicon, sin IMU):

- Tecleo: estadísticas de IKI (y DT=0 explícito) en la ventana N=50.
- Firma *proxy*: descriptores agregados de la serie T=128.
- Scroll: vector 6-D ya extraído.
- Par: `[template, query, |template−query|, template⊙query]` (+ flags de presencia en 3-mod). Template = media de las k ventanas de enrolamiento.
- 3-mod: rasgos concatenados, `pair=min`.

**L3 — RF como meta de scores:** `RandomForestClassifier` sobre el triplete (tecleo, firma, scroll) + máscaras de los expertos ya entrenados. Encodes congelados. Comparación directa con `ensemble_meta_3mod`.

L2 (RF sobre embeddings) no se ejecutó.

## Resultados eval leftover

EER en tanto por uno. Fuente RF: `data/rf_run/index.json`. Fuente propuesta: `data/eval_run/tables.json`.

| Sistema | Nivel | EER | AUC | FAR@1%FRR | Genuinos | Impostores |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RF tecleo | L1 | 0.320 | 0.767 | 0.738 | 2278 | 38726 |
| RF 3-mod (rasgos) | L1 | 0.309 | 0.756 | 0.919 | 2278 | 38726 |
| RF meta de scores | L3 | **0.303** | 0.762 | 0.583 | 2278 | 38726 |
| Ensemble meta 3-mod | — | **0.290** | 0.796 | 0.904 | 2278 | 38726 |
| Ensemble solo tecleo (z reclamado) | — | 0.304 | 0.773 | 0.905 | 2278 | 38726 |
| Unimodal tecleo ensemble | — | 0.476 | ~0.53 | — | 2278 | 38726 |
| Transformer 3-mod | — | **0.246** | 0.830 | 0.460 | 545 | 9265 |

Punto operativo con τ_dev (no es el EER de la tabla):

| Sistema | τ_dev | FAR eval | FRR eval |
| --- | ---: | ---: | ---: |
| RF tecleo | 0.650 | 0.0003 | 0.996 |
| RF 3-mod rasgos | 0.745 | 0.000 | 1.000 |
| RF meta L3 | 0.576 | 0.343 | 0.213 |

En L1 las probabilidades del bosque no se calibran entre *dev* y *eval*: τ_dev es casi un rechazo total. El ranking de sistemas debe leerse por **EER eval**, no por ese punto operativo. En L3 el τ_dev sí produce un compromiso usable (FAR 0.34 / FRR 0.21), peor EER que el meta neuronal.

TTD (k=2 ventanas consecutivas, eval): RF tecleo detecta en 300/306 secuencias; RF-meta en 306/306. Esas tasas altas van con umbrales de EER del split eval, no con τ_dev, y no se interpretan como TTD de reloj.

## Lectura

1. En este leftover press-only, una cabeza clásica **no** sustituye al Transformer cross-modal.
2. Concatenar firma *proxy* y scroll al RF de tecleo baja poco el EER (0.320 → 0.309); el canal extra no discrimina.
3. Un RF sobre los mismos scores de expertos no mejora al meta actual (0.303 vs 0.290).
4. Frente al unimodal de tecleo, el RF de rasgos sí es un verificador más fuerte (0.320 vs 0.476).
5. Cariello 1,5 % (tecleo 1 s + Vicon, split de sesiones en lab) **no** es este número y **no** va en F7.

IMU y mocap no entran en la tabla del sistema propuesto ni en F7.

## Archivo

`data/rf_run/` (índice y JSON versionables, sin caracteres):

| Ruta | Contenido |
| --- | --- |
| `index.json` | Protocolo + filas L1/L3 (EER, AUC, n pares) |
| `env.json` | Python, torch, seed, UTC |
| `rf_keystroke/` | Dev+eval L1 tecleo (`*_summary.json`, DET CSV/JSON, scores `.npz`) |
| `rf_3mod_features/` | Dev+eval L1 3-mod rasgos |
| `rf_meta_3mod/` | Dev+eval L3 RF-meta |

Figura: `python -m bmpid.eval.figures` escribe `eer_rf_vs_propuesto` (PNG+EPS) en `data/figures/` y en `project_tex/.../files/figures/cap06/`. Paleta Fase 7, EER en %, nota 545 vs 2278, **sin** 1,5 % de Cariello en el eje.
