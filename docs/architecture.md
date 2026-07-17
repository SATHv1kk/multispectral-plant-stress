# Architecture

This project contains **three** distinct networks. They are easy to confuse, and
the differences matter, so this page states exactly what each one is.

| | Single-frame FiLM | Temporal BiGRU | Demo |
|---|---|---|---|
| Module | `models/two_stream_film.py` | `models/temporal_bigru.py` | `demo/app.py` |
| Backbone | EfficientNet**V2**-B3 | EfficientNet-B3 (**v1**) | EfficientNet**V2**-B0 |
| Input | 1 frame, 320² RGB + 320²×4 indices | 5 frames, 320² RGB + indices | 1 frame, 224² RGB |
| Fusion | FiLM-style gating | concatenation | none (single stream) |
| Outputs | `gsw`, `Tleaf` | `Tleaf` only | `gsw`, `Tleaf` |
| Params | 14,093,504 | 13,959,600 | 6,083,538 |
| Status | thesis architecture | **released checkpoints** | live demo |

## 1. Single-frame two-stream, FiLM-style gating

This is the architecture described in the thesis and on the project page.

```
RGB  [320,320,3] ──> EfficientNetV2-B3 ──> GAP ──> LayerNorm ──> f_rgb (1536)
IDX  [320,320,4] ──> indices CNN       ──> GAP ──> LayerNorm ──> f_idx (96)

gate = sigmoid(W · f_idx)          # spectral stream gates the RGB features
f̃    = gate ⊙ f_rgb
z    = MLP(concat(f̃, f_idx))       # 512 → gelu → dropout(0.35) → 192 → gelu
     ├─> gsw    (z-scored log(gsw + ε))
     └─> Tleaf  (z-scored °C)
```

### On the name "FiLM"

Full FiLM (Perez et al., 2018) is an **affine** modulation:

```
f̃ = γ(c) ⊙ f + β(c)
```

This implementation learns the **multiplicative term only**, via a sigmoid gate,
and omits the additive `β`. It is therefore FiLM-*style* gating, not textbook
FiLM. The project page renders the general form `F̃ = Γ ⊙ F_RGB + Β`; the code
implements `F̃ = Γ ⊙ F_RGB`. The gate is a sigmoid, so `Γ ∈ (0,1)` — it can
suppress an RGB feature but never amplify or sign-flip it.

### Why gate RGB with the indices, and not the reverse?

The index map is a hand-derived physical quantity (NDVI/NDRE), so it is the
more trustworthy, lower-variance signal. Letting it decide *which* RGB features
survive uses it as a prior over appearance. Gating in the other direction would
let raw appearance — the thing most affected by illumination drift — veto the
physics.

## 2. Temporal two-stream BiGRU — the released checkpoints

**This is what `models/best_img_imgonly_seed{42,123,777}.keras` actually
contain.** It is not the single-frame model above.

```
rgb_seq [5,320,320,3] ──TimeDistributed(EfficientNet-B3, avg pool)──> [5,1536]
idx_seq [5,320,320,4] ──TimeDistributed(indices CNN)───────────────> [5,  96]
                        concat ─────────────────────────────────────> [5,1632]
                        Dropout(0.2)
                        Bidirectional(GRU(256))                     ─> [512]
                        Dense(384, gelu) → Dropout(0.3) → Dense(1)  ─> Tleaf
```

Three things differ from the thesis architecture, all deliberate:

1. **Backbone is EfficientNet-B3 v1, not V2-B3.** Verified against the weights:
   the checkpoint backbone has **497 weight tensors / 10,783,535 params**, an
   exact shape-for-shape match with `EfficientNetB3` and a mismatch with
   `EfficientNetV2B3` (536 tensors / 12,930,622 params). The checkpoint also
   contains `block7a`/`block7b` and a depthwise `block1a_dwconv`, neither of
   which exists in V2-B3 (V2 has no block 7 and uses *fused* convolutions in its
   early blocks). Do not "correct" this to V2 — the released weights would stop
   loading.
2. **Fusion is plain concatenation, not FiLM.** With a BiGRU downstream, the
   recurrent layer can learn cross-stream interaction itself, so the gate was
   redundant.
3. **Tleaf head only.** No gsw output is claimed, because gsw could not be
   regressed reliably from this dataset. See `results.md`.

### Loading them

`load_model()` fails on Keras ≥ 3.15 with
`ValueError: Cannot convert '5' to a shape` — a Keras regression in
deserializing the saved `TimeDistributed` config. The weights are fine. Use:

```python
from plant_stress.models.temporal_bigru import load_checkpoint
model = load_checkpoint("models/best_img_imgonly_seed42.keras")
```

which tries `load_model()` first and falls back to rebuilding the graph and
calling `load_weights()`, skipping config deserialization entirely.

## 3. Demo model

`EfficientNetV2-B0 → GAP → Dropout → Dense(128) → Dense(2)` at 224², outputting
`[gsw, Tleaf]`. RGB only — no spectral streams, no temporal context.

The backbone carries its own `Rescaling` + `Normalization` layers
(`include_preprocessing=True`), so it takes pixels in **[0, 255]**.
`efficientnet_v2.preprocess_input` is a documented pass-through and is safe to
call, but normalising by hand would scale the input twice.

The output head is unconstrained and linear, so `gsw` can come out **negative**
on an out-of-distribution photo. `app.py` surfaces that instead of clamping it —
a negative stomatal conductance is physically impossible and is the clearest
available signal that the image is nothing like the training data.

Verified against ground truth: on 98 real farmbed frames paired with LI-600
readings, this model scores **MAE 2.075 °C / RMSE 2.59** for `Tleaf` — matching
the reported ≈2.1 °C. `gsw` scores R² = −3.81, confirming it is not usable.
(Those frames likely overlap the training set, so this is a reproduction check,
not a clean validation.)

## Input scaling — the easiest way to break this project

Every model here is preceded by preprocessing, and each one wants a **different
input range**. Getting this wrong fails silently: no error, just bad numbers.

| Model | Backbone built with | Feed it |
|---|---|---|
| `two_stream_film` | `include_preprocessing=False` | ImageNet-normalised, via `indices.normalize_rgb` |
| `temporal_bigru` | v1 — always preprocesses | **[0, 255]** raw |
| demo | `include_preprocessing=True` | **[0, 255]** raw |

Measured cost of getting it wrong: feeding the demo model `[0, 1]` instead of
`[0, 255]` moves Tleaf MAE from **2.08 °C to 15.96 °C**. Every pixel maps to
roughly −2.1 after the internal `Rescaling(1/255)`, so the whole image reads as
black and the prediction collapses to the training mean (pred sd drops from 2.92
to 0.08).

`two_stream_film` disables the built-in preprocessing on purpose, so that
masking can happen *after* normalisation. Masked background then lands on
exactly `0.0`. If the backbone normalised internally instead, masked pixels
would arrive as `0` and be mapped to ≈ −2.1 — the network would read the
background as a saturated black leaf rather than as "nothing here".

For `temporal_bigru`, the true training-time scaling is **not recoverable** (the
training cell was never exported), so its `[0, 255]` contract is an assumption
from the architecture, not a verified fact. See `models/README.md`.

## Target transforms

`gsw` is trained as z-scored `log(gsw + ε)`. It is strictly positive and
right-skewed; the log makes residuals roughly symmetric, which is the assumption
Huber loss is built on. `Tleaf` is trained as a plain z-score. Invert both with
`two_stream_film.TargetScaler.denormalize`.

The scaler is fit on the **training split only**. Fitting on all rows would leak
validation statistics into training.
