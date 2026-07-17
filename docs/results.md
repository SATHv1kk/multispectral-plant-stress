# Results

Reported honestly: one target works, one does not.

## Leaf temperature (Tleaf) — working

| Metric | Value |
|---|---|
| MAE | **≈ 2.1 °C** |
| RMSE | ≈ 2.6 °C |
| R² (raw) | ≈ 0.199 |
| Heat-stress classification | strong (see below) |

Ablations, both favouring fusion:

- RGB + spectral fusion **beats** RGB-only
- RGB + spectral fusion **beats** indices-only

Tleaf is the reliable, physiologically grounded output — usable as a
non-destructive screen for early heat stress on the controlled farmbed.

### Heat-stress classification (test set)

|  | pred Low | pred Normal |
|---|---|---|
| **true Low** | 0 | 6 |
| **true Normal** | 0 | 52 |

Strong separation of Normal vs stressed leaves overall. Read the table
carefully, though: the classifier predicted `Normal` for **every** sample. It
scores well because the split is ~90% Normal, and it caught 0 of 6 Low-stress
leaves. On this validation split the heat classifier has not demonstrated
sensitivity to the minority class — accuracy here is mostly class prior, not
skill. The regression MAE is the more informative number.

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
