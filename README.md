# Multispectral Analysis of Plant Physiological Stress

**A two-stream CNN that reads crop stress from imagery — predicting leaf
temperature to ≈ 2.1 °C MAE.**

[**▶ Try the live demo — sathvik.info**](https://sathvik.info) &nbsp;·&nbsp;
[Architecture](docs/architecture.md) &nbsp;·&nbsp;
[Data pipeline](docs/data_pipeline.md) &nbsp;·&nbsp;
[Results](docs/results.md)

`EfficientNet` · `FiLM fusion` · `NDVI / NDRE` · `TensorFlow` · Maynooth University

---

## The problem

Drought and heat depress oat and barley yields long before any visible symptom
appears. The gold-standard readings — leaf temperature (`Tleaf`) and stomatal
conductance (`gsw`) — come from a handheld LI-600 porometer: **one leaf, one
number, one operator.**

That does not scale. This project asks whether the same physiological signal can
be recovered non-invasively from imagery a gantry can capture at plot scale,
every day, without touching a plant.

## What works, and what doesn't

| Target | Status | Result |
|---|---|---|
| **Leaf temperature** (`Tleaf`) | ✅ working | MAE ≈ **2.1 °C**, RMSE 2.6, R² 0.1986 |
| **Stomatal conductance** (`gsw`) | 🚧 in progress | R² −0.0163 — **no accuracy claimed** |

Fusion beats the indices-only MLP baseline (2.1 °C vs 2.5 °C). An **RGB-only
ablation was never run** — the thesis states it as a hypothesis, not a result.

`gsw` is physiologically volatile: stomata open and close over minutes in
response to light, water potential and VPD, and a single static frame does not
carry that state. It is reported for transparency, not as a validated result.

A VPD-leaf sensor-assisted benchmark scored R² ≈ 0.9955 for `Tleaf`. **That is
data leakage, not a result** — the LI-600 computes `VPDleaf` *from* `Tleaf`. It
is excluded from every claim here. See [results](docs/results.md).

## Quick start

```bash
git clone https://github.com/SATHv1kk/multispectral-plant-stress
cd multispectral-plant-stress
pip install -r requirements.txt
```

The live demo runs entirely client-side in the browser at
[sathvik.info](https://sathvik.info) — no server, no API.

### Use the research checkpoints

Weights are GitHub Release assets (267 MB each, over GitHub's file limit):

```bash
gh release download v1.0 --dir models/
```

```python
from plant_stress.models.temporal_bigru import load_checkpoint

model = load_checkpoint("models/best_img_imgonly_seed42.keras")
```

> **Note:** `tf.keras.models.load_model()` fails on these checkpoints with
> Keras ≥ 3.15. `load_checkpoint()` handles it — see [models/README.md](models/README.md).

### Train

```bash
python -m plant_stress.train \
  --csv data_with_indices.csv \
  --frames-root /path/to/OD1 \
  --seeds 42 123 777 \
  --mixed-precision
```

## How it works

```
Capture ──> Align ──> Indices + mask ──> Two-stream model ──> Calibrate
 3 streams   LI-600      NDVI, NDRE      EfficientNet + CNN    Ridge (no-leak)
 RGB/NIR/RE  timestamps  NDVI > 0.15     FiLM-style gating     3-seed + flip TTA
```

1. **Capture** — gantry rig records synchronised RGB, NIR and red-edge video
   over oat/barley plots under scheduled drought and heat treatments.
2. **Align** — LI-600 readings are paired to the nearest video frame by
   timestamp.
3. **Indices & mask** — NDVI and NDRE per pixel; `NDVI > 0.15` masks foreground
   vegetation and drops soil/background.
4. **Infer** — two-stream inference, 3-seed ensembling, horizontal-flip TTA,
   Ridge calibration on `Tleaf`.

**A note on splits, because it changes how you read the numbers.** The thesis's
reported results use a **random 80/20 split** (`random_state=42`, N=62 train /
N=16 test). This repo's `data/dataset.py` implements a **date-blocked** split
(last 25% of capture dates) instead, since frames from one day are
near-duplicates and a random split flatters the score. The date-blocked pipeline
is the sounder methodology but will **not** reproduce the thesis figures — see
[docs/results.md](docs/results.md).

## Three models, and which is which

This repo contains three networks. They are easy to confuse:

| | Module | Backbone | Input | Outputs | Params |
|---|---|---|---|---|---|
| **Single-frame FiLM** | `models/two_stream_film.py` | EfficientNet**V2**-B3 | 1 frame + indices | `gsw`, `Tleaf` | 14.1M |
| **Temporal BiGRU** | `models/temporal_bigru.py` | EfficientNet-B3 (**v1**) | 5 frames + indices | `Tleaf` | 14.0M |
| **Demo** | client-side ([sathvik.info](https://sathvik.info)) | EfficientNet**V2**-B0 | 1 frame, RGB only | `gsw`, `Tleaf` | 6.1M |

The **released checkpoints are the temporal BiGRU model**, not the single-frame
FiLM model the thesis describes, and their backbone is EfficientNet-B3 **v1**.
Both facts were verified against the weights themselves (497 tensors /
10,783,535 params — an exact match for `EfficientNetB3`, a mismatch for
`EfficientNetV2B3`). Details and caveats: [docs/architecture.md](docs/architecture.md).

## Stress thresholds

Project-specific interpretive bands — **not** universal agronomic standards.

| Drought (`gsw`, mol·m⁻²·s⁻¹) | | Heat (`Tleaf`, °C) | |
|---|---|---|---|
| Severe | < 0.05 | Normal | < 25 |
| High | 0.05 – 0.10 | Low | 25 – 28 |
| Medium | 0.10 – 0.20 | Medium | 28 – 30 |
| Low | 0.20 – 0.30 | High | 30 – 35 |
| Normal | > 0.30 | Severe | ≥ 35 |

Defined once in `src/plant_stress/config.py` so the thesis, this repo and the
live demo cannot drift apart.

## Layout

```
src/plant_stress/
  config.py           paths, geometry, stress thresholds
  indices.py          NDVI/NDRE + vegetation mask
  data/
    label_licor.py    LI-600 exports -> labelled ground truth
    dataset.py        tf.data pipelines, date-blocked split (see note above)
  models/
    two_stream_film.py  single-frame, FiLM-style gating  (thesis architecture)
    temporal_bigru.py   5-frame BiGRU, Tleaf only        (released weights)
  train.py            two-phase warm-up -> fine-tune
  predict.py          flip TTA + seed ensembling
  calibration.py      leakage-free Ridge calibration
  evaluate.py         metrics + stress confusion matrices
docs/                 architecture, data pipeline, results
notebooks/            original 20k-line Colab export (provenance)
models/               research checkpoints (via GitHub Release)
```

## Honest framing

The dataset is deliberately small — one controlled Maynooth farmbed, oats and
barley. This is a research prototype for a physiological signal, not a
production agronomic instrument, and it is framed that way throughout.

## Impact

A camera on a gantry — no probes, no operator per leaf — that reliably ranks
plants by heat stress makes early-warning irrigation and breeding trials
tractable at plot scale.

**Next:** temporal NPZ clip modelling so the network watches stomata respond over
time; VPD + ambient sensor fusion (leakage-free); larger, balanced datasets
across more cultivars.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE) — Sathvik, Maynooth University.
