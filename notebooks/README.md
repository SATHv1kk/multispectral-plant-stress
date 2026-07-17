# Notebooks

## `colab_workspace_full.py`

The original Google Colab workspace, exported verbatim (20,449 lines, ~404
cells). Kept **unedited** for provenance — it is the raw record of the
exploration, including dead ends, superseded runs and paths that only exist in
one Drive account.

It is not runnable as-is. `src/plant_stress/` is the cleaned, tested extraction
of the parts that matter. Use that.

Useful landmarks if you need to trace a result back to its source cell:

| Lines | What |
|---|---|
| ~6688 | EffNetV2-B3 + indices CNN + FiLM gating, 3 seeds (42/123/777) |
| ~8266 | "gswboost" run — gsw up-weighted, log-space target, 2 seeds |
| ~12429 | temporal clip model, TimeDistributed + attention pooling |
| ~13002 | 3-seed TTA ensemble inference over `best_img_imgonly_seed*.keras` |
| ~18960 | leakage-free Ridge calibration of Tleaf (the `_NOLEAK` calibrator) |
| ~20120 | thesis figure generation |

**Provenance gap:** no cell in this file trains
`best_img_imgonly_seed*.keras` — the released checkpoints. The notebook only
loads them. The only checkpoint it writes is `best_img_gswboost_seed{seed}.keras`.
The training cell for the released weights lived in a Colab session that was
never exported. See `models/README.md`.
