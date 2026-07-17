# Data pipeline

Capture → align → index → mask → model.

## 1. Capture

A gantry-mounted multispectral rig runs over oat and barley plots on the
Maynooth University farmbed, recording three synchronised 10 fps video streams:

| File | Band | What it sees |
|---|---|---|
| `rgb.mkv` | visible RGB | what the eye sees |
| `left.mkv` | near-infrared (NIR) | cell structure, water content |
| `right.mkv` | red-edge | chlorophyll, photosynthetic capacity |

Plots run under scheduled drought and heat treatments. Sampling follows a hybrid
design (`02_data/sampling_design_15x19.pdf` in the working archive): a
**constant** group measured every session to track change over time, plus a
**random** group rotated across days for diversity.

## 2. Ground truth

A handheld **LI-600 porometer/fluorometer** gives the reference physiology, one
leaf at a time:

| Channel | Meaning | Unit |
|---|---|---|
| `gsw` | stomatal conductance | mol·m⁻²·s⁻¹ |
| `Tleaf` | leaf temperature | °C |
| `Fv/Fm` | max PSII quantum yield | ~0.83 healthy, <0.75 stressed |
| `PhiPS2`, `ETR` | operating yield, electron transport | — |

Only `gsw` and `Tleaf` are regression targets. The fluorescence channels support
the descriptive statistics in the thesis.

This is exactly the bottleneck the project exists to remove: one leaf, one
number, one operator. It does not scale to plot level.

## 3. Alignment

Each capture day has a `timestamps.csv` giving a universal timestamp `ts` per
`frame_idx` per stream. LI-600 readings carry their own timestamps, so each
reading is paired to the nearest video frame.

**Rows whose three streams do not all resolve are dropped**, not imputed. See
`data/dataset.py::resolve_triplets`. Filenames drifted across capture days
(`frame_27.png`, `frame_0027.png`, `..._0027.jpg`), so resolution falls back
through: exact name → other extension → any file carrying the same `frame_<n>`
token.

## 4. Spectral indices

Computed per pixel (`indices.py`):

```
NDVI = (NIR − Red)     / (NIR + Red     + ε)
NDRE = (NIR − RedEdge) / (NIR + RedEdge + ε)
```

NDVI keys on chlorophyll absorption in red, so it separates plant from soil.
NDRE uses red-edge, which saturates much later than NDVI in dense canopy and so
keeps varying after NDVI has flattened.

The 4-channel index map is `[NDVI, NDRE, z(NIR), z(RedEdge)]`.

**NIR and red-edge are z-scored per image, not globally.** The rig has no
radiometric calibration target, so absolute band values drift with illumination
between capture days. Without per-image standardisation the network can learn
"which day is this" — a date shortcut that correlates with treatment schedule —
instead of "how stressed is this plant".

## 5. Foreground mask

```
mask = NDVI > 0.15
```

Applied to **both** streams with the same mask, so the RGB pixel at (i, j) and
the index vector at (i, j) always describe the same piece of leaf. This drops
soil and background before the network sees the frame.

## 6. Splitting

Two different splits exist in this project. Do not conflate them.

| | Thesis (reported results) | This repo (`data/dataset.py`) |
|---|---|---|
| Method | random 80/20, `random_state=42` | date-blocked, last 25% of dates |
| Size | N=62 train / N=16 test | varies with capture dates |
| Used for | every number in `results.md` | `train.py` |

**The thesis used a random 80/20 row split.** That is what produced the
≈2.1 °C MAE.

**`data/dataset.py` implements a date-blocked split instead**, via
`date_blocked_split()`. Frames from one day share illumination, canopy state and
treatment stage, so a random row split puts near-duplicates on both sides and
flatters the score. Holding out the *latest* dates also makes validation a
forward-in-time estimate, which is how the model would actually be used.

The date-blocked split is the more defensible methodology and came later, in the
notebook's post-thesis cells. It is **not** how the reported figures were
measured, and it would be expected to score worse. If you re-train with
`train.py` and get numbers below the thesis's, that is the split talking, not a
broken pipeline.

The dataset is small and deliberately so: a controlled farmbed trial. This is a
research prototype for a physiological signal, not a production instrument.

## 7. Augmentation

- **Horizontal flip** — applied to RGB, indices and mask together. For clips the
  flip decision is made once per clip; flipping some frames and not others would
  destroy the temporal continuity the BiGRU exists to read.
- **Brightness / contrast jitter — RGB only.** Perturbing NIR or red-edge would
  corrupt the physics baked into NDVI/NDRE.
