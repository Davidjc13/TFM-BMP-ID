# Ensemble NYIT (Fase 3)

Qué se implementó y ejecutó sobre `nyit_smartphone_dataset_home_lab_cleaned/` (tensores de `data/PREPROCESAMIENTO.md`). Diseño: `project/05-enfoque-ensemble.md`. Este documento es el **registro de ejecución** para pasar a LaTeX.

Reproducir:

```bash
python -m bmpid.train.experts --config configs/default.yaml
python -m bmpid.train.fusion --config configs/default.yaml
python -m bmpid.eval --config configs/default.yaml --model ensemble
```

Salida versionable: `data/ensemble_run/` (scores `.npz`, DET, inventario, TTD, `train_log.json`, `fusion_log.json`). Pesos: `checkpoints/ensemble/{keystroke,signature,scroll,fusion}.pt` (gitignore). IMU **no** entra.

Tests: `tests/test_ensemble.py` (forward BiLSTM / CNN-LSTM / MLP; Triplet en 2 usuarios sintéticos; fusión `omit` renormaliza; el módulo de train no importa IMU).

Mismo `splits.json` y mismo `eval` leftover que el Transformer (`data/TRANSFORMER.md`), con una diferencia de unidad de scroll (abajo).

---

## Por qué este corpus

NYIT comparte `user_id` en tecleo y swipe. La firma es `proxy_touch` (no firma legal). Tecleo y swipe son trials distintos: la fusión es **a nivel de usuario**, no multimodal simultáneo.

El experto de scroll usa el vector de gesto 6-D (`scroll.npz`), no los tokens por trial del Transformer. En eval, cada usuario tiene el mismo número de gestos de firma proxy y de scroll (41425 ítems; leftover 41335). El cuello de botella de `pair=min` es el **tecdeo** (2278 leftover), no el trial de swipe.

DT de tecleo es idénticamente 0 (solo press).

---

## Expertos ejecutados

Cada uno se entrena **aparte** con Triplet (margen 0.2) y mining semi-hard en el batch. Embeddings L2, score \(s=-d(\bar e_{\mathcal S}, q)\). \(k=5\). Z-score de consulta = estadísticos de enrolamiento del usuario reclamado (fusión) o del dueño de la sonda (unimodal).

| Experto | Arquitectura | Input | Fichero |
| --- | --- | --- | --- |
| Tecleo | BiLSTM 2 capas, hidden 64 → \(d=128\) | `[B, 50, 3]` | `src/bmpid/models/keystroke_lstm.py` |
| Firma proxy | CNN-1D (32 ch, k=5) + BiLSTM hidden 64 | `[B, 128, 6]` | `src/bmpid/models/signature_cnn_lstm.py` |
| Scroll | MLP 64-32-\(d\), dropout 0.3 | `[B, 6]` | `src/bmpid/models/scroll_mlp.py` |
| Triplet | `max(0, m + d(a,p) - d(a,n))` | — | `src/bmpid/losses/triplet.py` |

AdamW `lr=1e-3`, 12 épocas × 48 pasos, batch 8, seed 42, CPU (PyTorch 2.13+cpu). Mejor checkpoint por **EER en dev muestreado** (16 genuinos / 16 impostores por usuario), no por loss.

### Train (dev muestreado)

| Experto | Usuarios train | Época 1 EER | Mejor EER (época) | Loss final |
| --- | ---: | ---: | ---: | ---: |
| Tecleo | 40 | 0.364 | **0.277** (12) | 0.210 |
| Firma proxy | 49 | 0.412 | **0.387** (12) | 0.205 |
| Scroll | 49 | 0.430 | **0.410** (8) | 0.245 |

Curva: `data/ensemble_run/train_log.json`.

---

## Tres fusiones (expertos congelados)

Entrenadas **solo en dev** con pares leftover `pair=min`. Modalidad ausente: `missing_score: omit` (renormalizar pesos; no se rellena con 0 salvo flag).

| Estrategia | Cómo | EER dev (todas leftover min) | EER eval |
| --- | --- | ---: | ---: |
| Pesos fijos (config 0.34/0.33/0.33) | \(s=\sum w_m s_m\) | 0.416 | 0.347 |
| Pesos \(\propto 1/\mathrm{EER}\) unimodal en dev | 0.459 / 0.276 / 0.266 | 0.413 | 0.336 |
| Pesos aprendidos (softmax) | BCE en logits; **no** reabre expertos | 0.434 | 0.357 |
| **Meta-MLP** `[s, máscaras]` → 16-1 | BCE en dev | **0.326** | **0.290** |

El softmax aprendido carga sobre scroll (logit +1.40 vs firma −1.93) y **empeora** respecto a pesos fijos: no se usa como sistema propuesto. Se declara `fusion.strategy: meta` porque gana en **dev**.

EER unimodal **sobre los pares fused de dev** (no leftover completas): tecleo 0.256, firma 0.426, scroll 0.442.

---

## Evaluación (todas las leftover)

Galería \(k=5\) primeras muestras. Consultas = leftover. Impostores = todas las leftover de los demás. 16 usuarios dev / 18 eval. \(\tau\) de EER en dev (meta) aplicado en eval.

### Inventario eval

| Modalidad | Ítems | Enrolamiento (\(k=5\times 18\)) | Leftover | Uso en fusión 3-mod |
| --- | ---: | ---: | ---: | --- |
| Tecleo | 2368 | 90 | 2278 | las 2278 (`pair=min`) |
| Firma proxy | 41425 | 90 | 41335 | prefijo alineado 2278; resto en unimodal |
| Scroll (gestos) | 41425 | 90 | 41335 | igual que firma |

No se tiran gestos de scroll/firma: van a las columnas unimodales. Comparar EER fused 3-mod del ensemble (2278 vistas) con el Transformer fused 3-mod (545 vistas = trials de scroll tokens) **no es el mismo N**; sí es el mismo split de usuarios.

### Tabla para la memoria (eval, \(N=50\))

| Sistema | EER | AUC | FAR@1%FRR | n genuinos / impostores | Notas |
| --- | ---: | ---: | ---: | ---: | --- |
| **Meta** tecleo+firma+scroll | **0.290** | 0.796 | 0.904 | 2278 / 38726 | \(\tau_{\mathrm{dev}}\): FAR 0.470, FRR 0.047 |
| Fijo 1/EER | 0.336 | 0.732 | 0.746 | 2278 / 38726 | |
| Fijo igual | 0.347 | 0.719 | 0.796 | 2278 / 38726 | |
| Pesos softmax | 0.357 | 0.705 | 0.811 | 2278 / 38726 | peor que fijo |
| Ablación tecleo+firma (fijo) | 0.366 | 0.702 | 0.740 | 2278 / 38726 | peor que solo tecleo |
| Solo tecleo (z reclamado) | 0.304 | 0.773 | 0.905 | 2278 / 38726 | mismo enrolamiento fused |
| Unimodal tecleo (z owner) | 0.476 | 0.528 | 0.990 | 2278 / 38726 | |
| Unimodal firma proxy | 0.481 | 0.529 | 0.956 | 41335 / 702695 | ~azar |
| Unimodal scroll gesto | 0.495 | 0.505 | 0.992 | 41335 / 702695 | ~azar |

Dev meta: EER **0.326**, AUC **0.734**, 3489 genuinos / 52335 impostores.

### TTD (regla `consecutive` \(k=2\), 306 secuencias eval)

El umbral de EER fused y las ventanas TTD se ponen en **la misma escala**: cada ventana de tecleo se pasa por el fuser con firma/scroll ausentes (`omit`).

| Sistema | Detección | Media eventos | Mediana |
| --- | ---: | ---: | ---: |
| Meta 3-mod | 41.2 % | 24.2 | 2 |
| Fijo 1/EER | 40.5 % | 24.2 | 2 |
| Solo tecleo (z reclamado) | **98.0 %** | 5.8 | 2 |
| Unimodal tecleo (z owner) | 100 % | 7.1 | 5 |

La mediana 2 es el mínimo de la regla. El meta mejora EER pero **retrasa o pierde** detecciones TTD frente al experto de tecleo solo: el umbral fused no está calibrado para rachas unimodales.

### Cómo leer los números

- Proxy swipe **no identifica** en unimodal (~50 % EER), igual que en el Transformer.
- Ablación: tecleo+firma fijo (36.6 %) es peor que tecleo solo (30.4 %). Añadir scroll con **meta** baja a 29.0 %: beneficio pequeño y dependiente del fuser, no de un experto de scroll fuerte. Encaja la hipótesis del cap. 5 (scroll marginal) con el matiz de que el meta-clasificador sí extrae algo de las máscaras/scores conjuntos.
- Tecleo unimodal con z del dueño (47.6 %) es mucho peor que el mismo experto con z del reclamado en pares fused (30.4 %): hay que citar el protocolo de normalización, no solo el modelo.
- TypeNet-orden de magnitud: el EER de tecleo en dev *muestreado* (27.7 %) no se sostiene en leftover completas con z owner (47.6 %). El dump NYIT es press-only y N=50 con DT=0.

Transformer (mismo split, fusión 3-mod **otra unidad de scroll**): EER eval **0.246** / 545 vistas. Ensemble meta: **0.290** / 2278 vistas. Cifras sustituibles en tablas del cap. 5; no mezclar N sin nota.

Archivo: `data/ensemble_run/index.json`.

---

## Privacidad

- Train/eval no leen caracteres; los `.npz` ya no los contienen.
- Logs vía `format_safe`; checkpoints = `state_dict` + hiperparámetros.
- `privacy.store_raw_text: false`.
- `bmpid.train.experts` no importa IMU.

---

## Qué no hace este paso

- No reentrena el Transformer ni mezcla IMU.
- No usa semi-hard mining fuera del batch (negativos = otras anclas del minibatch).
- No declara el softmax-learned como propuesto (pierde en dev y eval).
- No sintoniza hiperparámetros en eval.
