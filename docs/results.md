# Results

All figures below are taken from the submitted thesis
(`Multispectral Analysis of Plant Physiological Stress Using Deep Convolutional
Neural Networks`, Sathvik, Maynooth University) and cross-checked against it.

Reported honestly: one target works, one does not.

## How the reported numbers were measured

**Random 80/20 split — `test_size=0.2`, `random_state=42` — giving N=62 train /
N=16 test.** Every headline number on this page comes from that split.

This matters, and the repo does not hide it: a random row split is **not** the
methodology `data/dataset.py` implements. That module does a **date-blocked**
split (last 25% of capture dates), because frames captured on the same day share
illumination, canopy state and treatment stage, so a random split puts
near-duplicates on both sides and flatters the score.

The date-blocked approach came later, in the notebook's post-thesis cells. It is
the more defensible methodology and it is what `train.py` uses — but it will
**not** reproduce the numbers below, and would be expected to score worse.
Treat the thesis figures as measured under the split the thesis states, not as
what the date-blocked pipeline would give.

## Leaf temperature (Tleaf) — working

| Metric | Value |
|---|---|
| MAE | **≈ 2.1 °C** |
| RMSE | ≈ 2.6 °C |
| R² (raw) | 0.1986 |
| Heat-stress classification | 88% accuracy, macro-F1 0.85 |

**Independently reproduced.** Running the shipped demo model over 98 real
farmbed frames paired with their LI-600 readings gives **MAE 2.075 °C / RMSE
2.59** — matching the thesis's ≈2.1 °C / 2.6 °C. (Those frames likely overlap
the training set, so this confirms the number reproduces; it is not a clean
held-out validation.)

Tleaf is the reliable, physiologically grounded output — usable as a
non-destructive screen for early heat stress on the controlled farmbed.

### Baselines — what was actually measured

| Model | Tleaf MAE | gsw MAE |
|---|---|---|
| Two-stream fusion | **≈ 2.1 °C** | ≈ 0.03 |
| Numeric-only MLP (NDVI + NDRE) | ≈ 2.5 °C | ≈ 0.04 |

Fusion beats the indices-only baseline. Note the thesis describes that MLP as a
fallback "used only when image frames were missing during early development" —
so it is a baseline of convenience, not a controlled ablation.

> **An RGB-only ablation was never run.** The thesis states fusion beating
> "RGB-only or indices-only baselines" once, in its *hypothesis*, and reports no
> RGB-only experiment. The project page lists "RGB + spectral fusion vs.
> RGB-only — Fusion wins" as a result; the submitted work does not support that
> claim. It remains untested.

### Heat-stress classification (test set)

The thesis reports **88% accuracy, macro-F1 0.85** on the N=16 test set
(Fig. 11, Confusion Matrix for Heat Levels).

> **Discrepancy worth resolving.** The project page publishes a heat confusion
> matrix of `true Low → [0, 6]`, `true Normal → [0, 52]`. That is 58 samples,
> not the thesis's 16, and it describes a classifier that predicted `Normal` for
> every input — which scores ~90% accuracy but macro-F1 ≈ 0.47, not 0.85. The
> two cannot both describe the same evaluation. The thesis figures are the ones
> reported here; the page's matrix appears to come from a different run and is
> not reproduced in this repo.

## Stomatal conductance (gsw) — in progress

**No gsw accuracy is claimed.** It is reported for transparency, not as a
validated result.

| Metric | Value |
|---|---|
| MAE | ≈ 0.03 mol·m⁻²·s⁻¹ |
| R² | ≈ −0.016 |

An R² below zero means the model is worse than predicting the training mean.

Why it fails: gsw is physiologically volatile. Stomata open and close over
**minutes** in response to light, water potential and vapour pressure deficit. A
single static frame does not carry that state. The dataset is also small and
imbalanced.

The `gsw` classifier remains under active development and is deliberately not
reported.

## The leakage result we are NOT claiming

A VPD-leaf sensor-assisted benchmark scored **R² ≈ 0.9955** for Tleaf.

**This is data leakage and is not a result.** The LI-600 computes `VPDleaf`
*from* `Tleaf`, so feeding it in hands the model a transformation of the answer.
It appears in the thesis as a diagnostic reference only, flagged transparently
there and excluded from the live demo and every headline claim.

`calibration.py` therefore excludes `VPDleaf` from its feature set by
construction. Future work moves to **ambient** VPD, which is leakage-free and
cheaper to instrument.

## Calibration

Raw predictions are slightly compressed toward the training mean — ordinary
regression-to-the-mean for a small-sample regressor. A ridge model on
`[pred, pred², ndvi, ndre]` corrects the scale.

Leakage rules, which is what the `_NOLEAK` suffix on the shipped calibrator
records:

- Fit on the **training split only**, applied unchanged to validation.
- Per-date calibrators only for dates present in **training** with ≥ 6 samples;
  everything else falls back to the global calibrator.
- `VPDleaf` is never a feature.

## Inference recipe for the reported numbers

3-seed ensemble (42 / 123 / 777), horizontal-flip TTA, then Ridge calibration on
Tleaf. Seeds are averaged **uniformly**: a weight search on a 16-row validation
set overfits the split, so the earlier weighted blends (e.g. `[0.05, 0.90,
0.05]`) explored in the notebook are not used for reported results.

## Honest framing

The dataset is deliberately small — one controlled Maynooth farmbed, oats and
barley. Everything here is a research prototype for a physiological signal, not
an agronomic-grade instrument, and the thresholds in `config.py` are
project-specific interpretive bands rather than universal agronomic standards.
