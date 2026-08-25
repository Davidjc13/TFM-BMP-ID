# Transformer NYIT (Fase 4)

Qué se implementó y ejecutó sobre `nyit_smartphone_dataset_home_lab_cleaned/` (tensores de `data/PREPROCESAMIENTO.md`). La especificación de diseño está en `project/04-enfoque-transformer.md`; este documento es el **registro de ejecución** para pasar a LaTeX.

Reproducir (todas las trazas, archivo en `data/transformer_run/`):

```bash
python -m bmpid.train.transformer --config configs/default.yaml
python -m bmpid.eval --config configs/default.yaml --archive-all
```

Salida versionable: `data/transformer_run/` (scores `.npz`, DET, inventario por usuario, TTD, `train_log.json`). Pesos: `checkpoints/transformer/best.pt` (gitignore). IMU **no** entra en este train.

Tests: `tests/test_transformer.py` (forward 1/2/3 modalidades; Set2set en 2 usuarios sintéticos; el módulo de train no importa `features.imu`).

---

## Por qué ahora y con este corpus

El checklist pide el Transformer **después** del ensemble unimodal. Aquí se adelanta la arquitectura A porque los tensores NYIT ya existen y el protocolo (splits usuario-disjuntos, \(k=5\), EER/AUC) es el mismo. El ensemble (Fase 3) sigue pendiente: el EER de esta fase **no** se compara aún con fusión de scores.

NYIT no alinea tecleo y swipe en el mismo trial (`signature.same_trial_as_keystroke: false`). La fusión es **a nivel de usuario**: una vista = una ventana de tecleo + un trazo proxy de firma + una secuencia de hasta 16 gestos de scroll, muestreada del mismo `user_id`. No es multimodal simultáneo.

La rama de firma es `proxy_touch` (swipe), no firma on-line legal. DT de tecleo es idénticamente 0 (solo press).

---

## Arquitectura ejecutada

Flujo (tres modalidades núcleo):

```
ventana_m → Linear(C_m → 128) + PE sinusoidal → Encoder Pre-LN (L=2, nhead=4, ff=256)
         → mean pooling enmascarado → token_m
tokens + CLS → MultiheadAttention (queries=CLS, keys/values=tokens)
         → Linear → e ∈ R^128, L2-normalizado
```

Si una modalidad falta: no se usa su secuencia; se inyecta un embedding `missing_m` aprendido en la fusión.

| Pieza | Fichero |
| --- | --- |
| Encoder + PE | `src/bmpid/models/encoders.py` |
| Cross-modal | `src/bmpid/models/cross_modal.py` |
| Sistema propuesto | `src/bmpid/models/behave_tf.py` |
| Set2set | `src/bmpid/losses/set2set.py` |
| Banco NYIT | `src/bmpid/train/dataset.py` |
| Minibatches \(k\)+pos+neg | `src/bmpid/train/sampler.py` |
| Train | `src/bmpid/train/transformer.py` |
| EER/AUC/FAR/TTD | `src/bmpid/eval/metrics.py` |
| Protocolo open-set | `src/bmpid/eval/protocol.py` |

Hiperparámetros (`configs/default.yaml`): `d_model=d_embed=128`, `nhead=4`, `L=2`, `ff=256`, dropout 0.1, `set2set_margin=0.2`, `k_enroll=5`, AdamW `lr=1e-3`, `batch_size=8`, 12 épocas × 48 pasos, seed 42, CPU (PyTorch 2.13+cpu).

**Pérdida** (no softmax de usuarios):

\[
\mathcal{L} = \max\bigl(0,\; m + d(\bar{e}_{\mathcal{S}}, q^+) - d(\bar{e}_{\mathcal{S}}, q^-)\bigr),\quad
\bar{e}_{\mathcal{S}}=\mathrm{L2}\bigl(\tfrac{1}{k}\sum_i e_i\bigr)
\]

\(d\) = euclídea sobre embeddings ya L2. Score de inferencia \(s = -d(\bar{e}_{\mathcal{S}}, q)\).

**Normalización en eval:** z-score por modalidad ajustado a las \(k\) ventanas de enrolamiento del usuario reclamado (no al global de train). El clip 1–99 ya está aplicado en los `.npz`.

---

## Criterios de “hecho” (código)

| Criterio (`04-enfoque-transformer.md`) | Resultado |
| --- | --- |
| Forward con 3, 2 o 1 modalidad sin error | `test_forward_one_two_three_modalities` |
| Set2set baja en overfit de 2 usuarios sintéticos | `test_overfit_two_synthetic_users` |
| Eval EER (no softmax) | `python -m bmpid.eval` escribe EER/AUC/FAR@1%FRR/TTD |

`pytest`: 42 tests OK tras esta fase.

---

## Entrenamiento (corrida)

49 usuarios de train (unión de modalidades; 40 tienen tecleo). Mejor checkpoint por **EER en dev**, no por loss.

| Época | Loss Set2set | EER dev | AUC dev |
| ---: | ---: | ---: | ---: |
| 1 | 0.0598 | **0.1465** | 0.9194 |
| 8 | 0.0435 | 0.2207 | 0.8995 |
| 12 | 0.0484 | 0.1523 | 0.9185 |

Se conserva la época 1 (`checkpoints/transformer/best.pt`). La loss sigue bajando un poco, pero el EER de dev no mejora: el margen Set2set en train no es la métrica oficial.

---

## Evaluación (todas las trazas residuales)

No se muestrean consultas. Galería: \(k=5\) primeras muestras por modalidad. Consultas = todo lo que queda (`leftover`).

**Fusión (`pair: min`):** por usuario, tantas vistas como la modalidad más corta (en NYIT, casi siempre scroll). Se usa el prefijo alineado de tecleo y de firma; el resto de trazas de firma/tecleo **sí se evalúa** en las corridas unimodales. Impostores: todas las consultas de todos los demás usuarios. Z-score de la consulta = estadísticos de enrolamiento del **usuario reclamado**.

**Unimodal:** todas las leftover de esa modalidad (41335 trazas proxy de firma en eval). Z-score del **dueño de la sonda** (`owner`) para poder embeber cada traza una sola vez.

TTD: todas las ventanas de tecleo leftover de cada impostor, regla `consecutive` \(k=2\).

16 usuarios dev / 18 eval. \(\tau\) de EER en **dev** (fusión) y se aplica en eval fusión.

### Inventario eval (dump completo)

| Modalidad | Ítems en split | Enrolamiento (\(k=5\times 18\)) | Leftover usadas |
| --- | ---: | ---: | ---: |
| Tecleo | 2368 | 90 | 2278 (unimodal y tecleo+firma) |
| Firma (proxy swipe) | 41425 | 90 | 41335 (unimodal); 545 en fusión 3-mod (tope scroll) |
| Scroll (trials) | 635 | 90 | 545 (fusión 3-mod y unimodal scroll) |

Detalle por usuario: `data/transformer_run/fused/eval_inventory.json`.

### Tabla para la memoria (eval, \(N=50\), todas las leftover)

| Sistema | EER | AUC | FAR@1%FRR | n genuinos / impostores | Notas |
| --- | ---: | ---: | ---: | ---: | --- |
| Fusión tecleo+firma+scroll | **0.246** | 0.830 | 0.460 | 545 / 9265 | \(\tau_{\mathrm{dev}}\): FAR 0.330, FRR 0.132 |
| Fusión tecleo+firma | 0.281 | 0.796 | 0.521 | 2278 / 38726 | min(tecleo, firma leftover) |
| Fusión solo tecleo (ramas masked, z reclamado) | 0.281 | 0.792 | 0.808 | 2278 / 38726 | mismo checkpoint |
| Unimodal tecleo (todas leftover, z owner) | 0.446 | 0.586 | 0.956 | 2278 / 38726 | no reentrenado |
| Unimodal firma proxy (todas leftover) | 0.502 | 0.496 | 0.989 | 41335 / 702695 | azar: el swipe no identifica solo |
| Unimodal scroll | 0.472 | 0.523 | 0.991 | 545 / 9265 | azar |

Dev fusión (todas leftover min): EER **0.180**, AUC **0.891**, 349 genuinos / 5235 impostores. TTD eval fusión: 306 secuencias, detección 97.7 %, media 11.6 eventos, mediana 2.

La corrida anterior con **16 consultas/usuario** (EER eval 22.2 %) queda como ablación de muestreo; la cifra oficial a citar es la de esta tabla (todas las trazas).

### Cómo leer estos números

- Fusión 3-mod (~25 % EER) usa 545 vistas = todos los trials de scroll. No tira trazas de scroll; firma y tecleo extra van a las columnas unimodales.
- Unimodal firma/scroll ~50 % EER: el modelo se entrenó fusionando ramas; el proxy swipe no basta como identidad. Documentar, no ocultar.
- Tecleo unimodal (todas las ventanas) es peor que la fusión 3-mod: el emparejado con scroll selecciona un subconjunto de ventanas y el z-score del reclamado ayuda.
- TTD mediana = 2 es el mínimo de \(k_{\mathrm{consecutive}}=2\); reportar media y tasa de detección.

Archivo: `data/transformer_run/index.json`. Scores y DET: `fused/*_scores.npz`, `fused/*_det.json`. Curva train: `data/transformer_run/train_log.json`.

---

## Privacidad

- Train/eval no leen caracteres; los `.npz` ya no los contienen.
- Logs vía `format_safe`; checkpoints = `state_dict` + hiperparámetros, no secuencias de eval.
- `privacy.store_raw_text: false` sigue validado en config.
- El módulo `bmpid.train.transformer` no importa IMU.

---

## Qué no hace este paso

- No entrena el ensemble (BiLSTM/Triplet, CNN-LSTM, MLP scroll ni fusores).
- No compara con BehaveFormer/HuMIdb (haría falta IMU + otro corpus).
- No reentrena un Transformer *solo* tecleo (la columna “solo tecleo” es el modelo de 3 ramas con firma y scroll masked).
- No sintoniza hiperparámetros en eval.

Siguiente: IMU opcional (Fase 6) si se replica BehaveFormer; las tablas ensemble vs Transformer ya están en `data/ENSEMBLE.md` y `data/TRANSFORMER.md`.
