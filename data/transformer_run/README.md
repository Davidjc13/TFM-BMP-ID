# Archivo de trazas — Transformer NYIT

Bundle versionable para análisis y memoria (sin caracteres ni xy crudos). Narrativa: [`../TRANSFORMER.md`](../TRANSFORMER.md).

Regenerar:

```bash
python -m bmpid.eval --config configs/default.yaml --archive-all
```

| Ruta | Contenido |
| --- | --- |
| `index.json` | Índice de corridas + entorno |
| `env.json` | Python, torch, seed, UTC |
| `train_log.json` | Loss y EER de dev por época |
| `fused/` | Dev+eval 3 modalidades y ablaciones; `*_scores.npz` (y, scores, user_id seudónimo), `*_det.json`, `*_inventory.json` (conteo de **todas** las leftover), `*_per_user.json`, `*_ttd.json` |
| `unimodal/` | Todas las leftover de tecleo / firma / scroll por separado |

Fusión `pair=min`: no se descartan trials de scroll; las trazas extra de firma (41335 leftover en eval) están en `unimodal/eval_no_keystroke_scroll_scores.npz`.
