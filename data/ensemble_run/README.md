# Archivo de trazas — Ensemble NYIT

Bundle versionable (sin caracteres ni xy crudos). Narrativa: [`../ENSEMBLE.md`](../ENSEMBLE.md).

Regenerar:

```bash
python -m bmpid.eval --config configs/default.yaml --model ensemble
```

| Ruta | Contenido |
| --- | --- |
| `index.json` | Índice de corridas + entorno + `best_fuser_dev` |
| `env.json` | Python, torch, seed, UTC |
| `train_log.json` | Loss y EER de dev por experto y época |
| `fusion_log.json` | EER de las cuatro fusiones en dev; pesos |
| `fused/` | Dev+eval 3-mod (meta oficial) y `eval_{fixed,fixed_inv_eer,learned,meta}`; ablaciones; `*_scores.npz` |
| `unimodal/` | Leftover de tecleo / firma / scroll por separado |

Fusión `pair=min`: el tope es leftover de tecleo (2278 en eval). Firma y scroll extra están en `unimodal/`.
