# Model weights

Research checkpoints are **not** in git — they are ~267 MB each, above GitHub's
100 MB hard limit. They ship as **GitHub Release assets**.

## Download

```bash
gh release download v1.0 --dir models/
```

Or grab them from the [Releases page](../../releases) and drop them here.

| File | Size | What it is |
|---|---|---|
| `best_img_imgonly_seed42.keras` | 267 MB | temporal BiGRU, seed 42 |
| `best_img_imgonly_seed123.keras` | 267 MB | temporal BiGRU, seed 123 |
| `best_img_imgonly_seed777.keras` | 267 MB | temporal BiGRU, seed 777 |
| `tleaf_calibrator_NOLEAK.pkl` | 1 KB | Ridge calibrator for Tleaf (sklearn) |

The distilled single-frame demo model (EfficientNetV2-B0, 26 MB) is deployed
client-side in the browser at [sathvik.info](https://sathvik.info) — it is not
served from this repository.

## What these checkpoints actually are

Verified by inspecting the weights, not by trusting the filename:

- **Temporal 5-frame BiGRU models that output `Tleaf` only.** They are *not* the
  single-frame FiLM model in `src/plant_stress/models/two_stream_film.py`.
- **Backbone is EfficientNet-B3 (v1), not EfficientNetV2-B3** — 497 weight
  tensors / 10,783,535 params, an exact match with `EfficientNetB3` and a
  mismatch with `EfficientNetV2B3` (536 / 12,930,622).
- Full model: **522 tensors / 13,959,600 params**.

`src/plant_stress/models/temporal_bigru.py` reconstructs this exactly; its
graph accepts these weights tensor-for-tensor.

The name `imgonly` means "image-only" — no sensor-derived inputs, i.e. the
leakage-free configuration. It does **not** mean RGB-only; these models take
both RGB and the spectral index map.

## Loading

```python
from plant_stress.models.temporal_bigru import load_checkpoint

model = load_checkpoint("models/best_img_imgonly_seed42.keras")
# {'rgb_seq': [B,5,320,320,3], 'idx_seq': [B,5,320,320,4]} -> [B,1] z-scored Tleaf
```

`tf.keras.models.load_model()` **fails on Keras ≥ 3.15** with
`ValueError: Cannot convert '5' to a shape`. That is a Keras regression in
deserializing the saved `TimeDistributed` config — the weights themselves are
fine. `load_checkpoint()` tries the direct load, then falls back to rebuilding
the graph and calling `load_weights()`, which skips config parsing entirely.

The checkpoints were written with **Keras 3.10 / TF 2.x**.

## Known gap: training code provenance

**The exact cell that trained these checkpoints is not in this repository.**

The Colab export (`notebooks/colab_workspace_full.py`, 20,449 lines) only ever
*consumes* `best_img_imgonly_seed*.keras` — it loads them by glob for TTA
ensembling and calibration. No `ModelCheckpoint` in that notebook writes them;
the only checkpoint it writes is `best_img_gswboost_seed{seed}.keras`. The
training cell lived in a Colab session that was not exported.

What that means in practice:

- The weights are usable and reproducible **for inference** — verified.
- `train.py` trains the **single-frame FiLM** model (`two_stream_film.py`),
  whose training code *is* fully present in the notebook. It will not reproduce
  these temporal checkpoints bit-for-bit.
- `temporal_bigru.build_model()` is an architecture reconstruction validated
  against the weights, not a transcription of the original training cell. Its
  hyperparameters (dropout 0.2/0.3, GRU 256, Dense 384) are read directly out of
  the saved config, so they are exact; the optimizer, LR schedule and epoch
  count used for that run are **not recoverable** from the checkpoint alone.

If the original notebook turns up in Drive, add it under `notebooks/` and this
note can go.

## Calibrator

`tleaf_calibrator_NOLEAK.pkl` is a pickled sklearn `Ridge` + `StandardScaler`.
It needs `scikit-learn` installed and is version-sensitive, as sklearn pickles
always are. `src/plant_stress/calibration.py` can refit an equivalent from
predictions if the pickle will not load — that is the more durable path.
