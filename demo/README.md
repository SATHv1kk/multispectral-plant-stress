---
title: Plant Stress Demo
emoji: 🌾
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
python_version: '3.10'
app_file: app.py
pinned: false
license: mit
short_description: Predicts leaf temperature and stress from a leaf photo
---

# Plant Stress Demo

Upload an oat/barley leaf photo and get predicted leaf temperature (`Tleaf`) and
stomatal conductance (`gsw`), each mapped to a stress band.

**Full project:** [sathvik.info](https://sathvik.info) ·
**Code:** [GitHub](https://github.com/SATHv1kk/multispectral-plant-stress)

## What this model is

A **lightweight, single-frame, RGB-only** variant — deliberately not the research
model:

| | Demo | Research model |
|---|---|---|
| Backbone | EfficientNetV2-B0 (6.1M params) | EfficientNet-B3 / V2-B3 (~14M) |
| Input | one 224² RGB photo | 320² RGB **+** NIR **+** red-edge |
| Spectral indices | none | NDVI, NDRE, z(NIR), z(RE) |
| Fusion | none — single stream | FiLM-style gating |
| Temporal | none | 5-frame clips (BiGRU variant) |

The full model needs a synchronised multispectral capture rig, not a phone
photo. That is the whole point of the demo being separate.

## Honest limits

- Predictions are **raw model output** — research-grade, not agronomic-grade.
- `Tleaf` is the trustworthy output. `gsw` is shown for transparency and is
  **not validated** (R² ≈ −0.016 on the research model).
- The output head is unconstrained, so `gsw` can be **negative** on an
  out-of-distribution photo. The app says so rather than clamping it — a
  negative stomatal conductance is physically impossible and is the clearest
  signal the image is nothing like the training data.
- Trained on a small controlled farmbed dataset (oats and barley, Maynooth
  University). Expect nonsense on anything else.

## Run locally

```bash
pip install -r requirements.txt gradio
python app.py
```

`app.py` runs unchanged locally and on Hugging Face ZeroGPU: the `spaces` import
is optional (no-op decorator when absent), and the model loader prefers
`saved_model/` when present, falling back to the committed `.keras` file.

## Regenerating the SavedModel

```bash
python export_saved_model.py
```

A `.keras` file stores layer *configs* that the loading Keras version must
re-parse, which can fail when a deployment pins an older Keras. A SavedModel
stores a frozen graph and replays without consulting those configs. The export
script verifies its output matches the source model before writing.

`saved_model/` is gitignored — regenerate it rather than committing 50 MB of
derived artifact.

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio interface + inference (flip TTA) |
| `rgb_singleframe_demo.keras` | the demo model (26 MB, committed) |
| `export_saved_model.py` | `.keras` → SavedModel, with verification |
| `requirements.txt` | Space runtime pins |
