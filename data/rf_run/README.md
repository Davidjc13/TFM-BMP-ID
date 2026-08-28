# Archivo de trazas — Random Forest leftover NYIT (Fase 8)

Bundle versionable (sin caracteres ni xy crudos). Narrativa de resultados: [`../RF.md`](../RF.md).

Comparación **en el protocolo propio** (no clona Cariello + Vicon). Regenerar:

```bash
python -m bmpid.eval.rf --config configs/default.yaml
python -m bmpid.eval.figures --config configs/default.yaml
```

| Sistema | EER eval | AUC | Genuinos |
| --- | ---: | ---: | ---: |
| RF tecleo (L1) | 0.320 | 0.767 | 2278 |
| RF 3-mod rasgos (L1) | 0.309 | 0.756 | 2278 |
| RF meta scores (L3) | 0.303 | 0.762 | 2278 |
| Ensemble meta 3-mod | 0.290 | 0.796 | 2278 |
| Transformer 3-mod | 0.246 | 0.830 | 545 |

El RF no replica el 1,5 % de Cariello (otro split, mocap, ventanas 1 s).
