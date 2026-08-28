# Archivo de trazas — Evaluación y privacidad (Fase 5)

Bundle versionable (sin caracteres ni xy crudos). Narrativa: [`../EVALUACION.md`](../EVALUACION.md).

Regenerar:

```bash
python -m bmpid.eval --config configs/default.yaml --seqlen
python -m bmpid.eval --config configs/default.yaml --report
```

| Ruta | Contenido |
| --- | --- |
| `index.json` | Puntero a tablas + `privacy_ok` + recuento de filas |
| `tables.json` | Filas oficiales Transformer / ensemble / seq_len + SOTA (no propias) |
| `env.json` | Python, torch, seed, UTC |
| `det/` | CSV DET (`threshold,far,frr`) exportados desde las corridas 3 y 4 y seq_len |
| `seqlen/n{25,50,100}/` | Eval leftover unimodal tecleo; N=50 reusa el experto oficial |

Los scores `.npz` de seq_len son pares genuino/impostor (IDs seudónimos), no texto.
