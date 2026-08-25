# Preprocesamiento NYIT (Fase 2)

Qué se hizo sobre `nyit_smartphone_dataset_home_lab_cleaned/`, por qué, y dónde quedan los tensores. La especificación de diseño está en `project/03-preprocesamiento.md`; este documento es el **registro de ejecución**.

Reproducir:

```bash
python -m bmpid.data --config configs/default.yaml
python -m bmpid.features --config configs/default.yaml
```

Salida: `data/processed/nyit/features/` (`.npz` en gitignore). IMU **no** se escribe salvo `--include-imu`.

---

## Por qué este corpus y este recorte

NYIT es el único dump activo con el mismo `user_id` en tecleo y swipe (home + lab, iPhone XR). Eso permite fusión a nivel de usuario sin emparejar corpora distintos.

No se usa como “firma legal”: `signature.decision: proxy_touch` sobre `swipe_data.csv`. Tecleo y swipe son **trials distintos** (`same_trial_as_keystroke: false`): no hay ventana multimodal simultánea.

IMU (`accelerometer.csv` / `gyroscope.csv`) está en el manifiesto de Fase 1 y el código `features/imu.py` existe para una réplica SOTA opcional. **No** entra en el train del sistema propuesto (`imu.enabled_in_proposed: false`; `bmpid.features` y `bmpid.train` no importan IMU en el camino por defecto).

---

## Decisiones que aplican a todas las modalidades

| Decisión | Por qué |
| --- | --- |
| Splits **usuario-disjuntos** (49 train / 16 dev / 18 eval, `seed: 42`) | Open-set: el embedding no puede ver al usuario de eval en el train. |
| Clip percentiles **1–99** y z-score **solo con train** | Evitar leakage: eval no define ni recortes ni μ/σ del modelo. |
| Tensores guardados = **ya recortados**, **sin** z-score aplicado | En eval, el z-score debe poder ser el del **enrolamiento** del usuario (`k_enroll`), no el global de train. `norm_stats.json` guarda `zscore_train` y `zscore_user_train`. |
| Sin caracteres en disco | RGPD / minimización: el CSV crudo tiene la tecla; el npz solo timings + `user_id` seudónimo. |
| Un punto por timestamp en swipe | El dump repite `(t,x,y)` casi idénticos; `dt≈0` hacía velocidades/aceleraciones explosivas. |

---

## Tecleo

**Crudo:** `keystrokes_data.csv` → pares `(epoch_ms, carácter)`. El carácter se descarta al parsear (solo se lee el timestamp).

**Limitación del dump:** no hay `keyup` / release. No se puede medir dwell.

**Codificación (compatible con el vector de 3 canales de la memoria):**

- \(IKI_i = t_{press,i+1} - t_{press,i}\) (segundos)
- \(DT_i = 0\) (dwell desconocido)
- \(FT_i = IKI_i\) (si dwell = 0, flight = IKI)

Mantener 3 canales permite enchufar el mismo encoder el día que un corpus sí tenga release; en NYIT DT es idénticamente 0 (clip 1–99 de train: `[0, 0]`).

**Ventanas:** `N = 50`, `stride = 25`. Si hay menos de 50 intervalos IKI, una ventana con padding 0 y máscara. Si hay más, sliding windows (se añade la última ventana alineada al final para no tirar cola).

**Salida:** `[10222, 50, 3]` + máscara `[10222, 50]`.

| split | ventanas |
| --- | ---: |
| train | 6428 |
| dev | 1426 |
| eval | 2368 |

IKI típico (train, tras clip): μ ≈ 0.35 s, p1 ≈ 54 ms, p99 ≈ 3.66 s. El clip recorta pausas largas (pensar, cambiar de campo) que no son el ritmo de tecleo.

---

## Scroll (gesto → vector)

**Crudo:** `swipe_data.csv` `(t, x, y)`, tareas `sit_and_swipe` / `walk_and_swipe` / `home/swiping`.

**Segmentación:** corte si el hueco entre puntos > `gesture_gap_ms: 300` (un swipe vs el siguiente). Gestos con &lt; 2 puntos se descartan.

**Vector de 6 D** (`scroll_gesture_to_vector`):

1. velocidad media  
2. velocidad máxima  
3. aceleración media  
4. longitud euclidiana del trazo  
5. duración (s)  
6. desviación angular respecto al eje vertical de pantalla `(0, 1)`

Velocidad y aceleración se calculan **tras interpolar el gesto a 100 Hz**, no con `Δt` crudo entre samples táctiles (esos `Δt` son irregulares y a veces ~0).

**Salida ensemble:** `[175239, 6]` (un vector por gesto).  
**Salida Transformer:** `[2749, 16, 6]` + máscara — hasta `max_gestures: 16` tokens por trial, padding si hay menos.

---

## Firma (proxy táctil)

Misma segmentación de gestos que scroll (mismo fichero). No hay presión (`pressure_available: false`) → **C = 6**, no 8.

Por gesto:

1. Coordenadas relativas al primer punto (invarianza a traslación).
2. Interpolación temporal a `fs: 100` Hz.
3. Longitud fija `T: 128` (recorte por interpolación si el trazo es más largo; pad + máscara si es más corto).
4. Canales: \(x, y, \Delta x, \Delta y\), velocidad, aceleración.

**Por qué T fijo:** el CNN-LSTM / encoder de firma necesita un tensor `[B, T, C]` comparable entre usuarios y duraciones.

**Salida:** `[175239, 128, 6]` + máscara. Casi 1:1 con los vectores de scroll (misma unidad de gesto).

---

## IMU (código listo, no ejecutado sobre el dump)

`imu_timestep_features` → 36-D por instante: acc+gyro (6), 1.ª y 2.ª derivadas (12), 3 bandas FFT locales × 6 canales (18). Alineación a timestamps de tecleo: `align_to_times`.

No hay `imu.npz` en esta corrida. Activar solo para réplica BehaveFormer:

```bash
python -m bmpid.features --config configs/default.yaml --include-imu
```

---

## Artefactos

| Fichero | Contenido |
| --- | --- |
| `data/processed/nyit/features/keystroke.npz` | `x`, `mask`, `user_id`, `split` |
| `data/processed/nyit/features/signature.npz` | igual, `T=128` |
| `data/processed/nyit/features/scroll.npz` | vectores de gesto |
| `data/processed/nyit/features/scroll_tokens.npz` | tokens por trial |
| `*_index.jsonl` | `user_id`, `session_id`, `split`, `start`/`end` en el npz (sin teclas ni xy) |
| `norm_stats.json` | percentiles de clip y z-score de **train** (global y por usuario de train) |

Código: `src/bmpid/features/{keystroke,signature,scroll,windowing,normalize,imu,extract}.py`.

Tests: `tests/test_features.py` (ventana N=50 sin caracteres; trazo vertical sintético → ángulo ≈ 0; clip no usa eval; train no importa IMU).

---

## Qué no hace este paso (siguiente fase)

No entrena modelos. El z-score de consulta en eval hay que aplicarlo con las *k* ventanas de enrolamiento de ese usuario, usando las funciones de `normalize.py` y `norm_stats.json` solo como fallback de train / ablación.
