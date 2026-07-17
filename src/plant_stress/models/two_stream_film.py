"""Single-frame two-stream network with FiLM-style gating.

This is the architecture described in the thesis and on the project page:

    RGB  [320,320,3] --> EfficientNetV2-B3 --> GAP --> LayerNorm --> f_rgb (1536)
    IDX  [320,320,4] --> indices CNN       --> GAP --> LayerNorm --> f_idx (96)

    gate = sigmoid(W f_idx)            # FiLM-style, multiplicative only
    f~   = gate * f_rgb                # spectral stream modulates RGB features
    z    = MLP(concat(f~, f_idx))
    -> gsw   (z-scored log gsw)
    -> Tleaf (z-scored Tleaf)

A note on the name, because the thesis figure and this code differ slightly.
Full FiLM (Perez et al., 2018) applies an affine transform, f~ = gamma * f + beta.
This implementation learns the multiplicative term only (gamma via a sigmoid
gate) and omits the additive beta. It is therefore FiLM-*style* gating rather
than textbook FiLM. The distinction is documented in docs/architecture.md.

Targets are learned in a transformed space:
  - gsw is trained as z-scored log(gsw + eps). gsw is right-skewed and strictly
    positive, so the log makes the residuals roughly symmetric, which is what
    Huber loss assumes.
  - Tleaf is trained as a plain z-score.
Use `denormalize_targets` to get back to physical units.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import models as M

from ..config import GSW_EPS, IMG_SIZE, INDEX_CHANNELS


def build_indices_cnn(x: tf.Tensor) -> tf.Tensor:
    """Compact CNN over the 4-channel index map.

    Deliberately small: the index map is already a hand-derived physical
    feature, so it needs far less capacity than raw RGB. Strided convs
    downsample rather than pooling layers, to keep the tower cheap.
    """
    for filters in (32, 64, 96):
        x = L.Conv2D(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("gelu")(x)
    x = L.GlobalAveragePooling2D()(x)
    return L.LayerNormalization(epsilon=1e-6)(x)


def build_model(img_size: int = IMG_SIZE):
    """Build the single-frame two-stream regressor.

    Returns:
        model: keras Model mapping {'rgb', 'indices'} -> {'gsw', 'Tleaf'}.
        backbone: the EfficientNetV2-B3 instance, exposed so the caller can
            freeze/unfreeze it between the warm-up and fine-tune phases.
    """
    rgb_in = L.Input(shape=(img_size, img_size, 3), name="rgb")
    idx_in = L.Input(shape=(img_size, img_size, INDEX_CHANNELS), name="indices")

    backbone = tf.keras.applications.EfficientNetV2B3(
        include_top=False, input_tensor=rgb_in, weights="imagenet"
    )
    f_rgb = L.GlobalAveragePooling2D(name="rgb_gap")(backbone.output)
    f_rgb = L.LayerNormalization(epsilon=1e-6, name="rgb_ln")(f_rgb)

    f_idx = build_indices_cnn(idx_in)

    # FiLM-style gate: the spectral stream decides which RGB features matter.
    gate = L.Dense(f_rgb.shape[-1], activation="sigmoid", name="film_gate")(f_idx)
    gated = L.Multiply(name="film_mul")([f_rgb, gate])

    z = L.Concatenate(name="fuse")([gated, f_idx])
    z = L.Dense(512)(z)
    z = L.Activation("gelu")(z)
    z = L.Dropout(0.35)(z)
    z = L.Dense(192)(z)
    z = L.Activation("gelu")(z)

    # float32 heads keep the outputs stable under mixed-precision training.
    gsw_out = L.Dense(1, name="gsw", dtype="float32")(z)
    tleaf_out = L.Dense(1, name="Tleaf", dtype="float32")(z)

    model = M.Model(
        inputs={"rgb": rgb_in, "indices": idx_in},
        outputs={"gsw": gsw_out, "Tleaf": tleaf_out},
        name="two_stream_film",
    )
    return model, backbone


class TargetScaler:
    """Converts gsw/Tleaf between physical units and the network's target space.

    Fit on the TRAINING split only. Fitting on all rows would leak validation
    statistics into training.
    """

    def __init__(self) -> None:
        self.gsw_mu = self.gsw_sd = self.tleaf_mu = self.tleaf_sd = None

    def fit(self, gsw: np.ndarray, tleaf: np.ndarray) -> "TargetScaler":
        gsw_log = np.log(np.clip(gsw, 0.0, None) + GSW_EPS)
        self.gsw_mu, self.gsw_sd = float(gsw_log.mean()), float(gsw_log.std() + 1e-8)
        self.tleaf_mu, self.tleaf_sd = float(tleaf.mean()), float(tleaf.std() + 1e-8)
        return self

    def normalize(self, gsw: np.ndarray, tleaf: np.ndarray):
        gsw_log = np.log(np.clip(gsw, 0.0, None) + GSW_EPS)
        return (gsw_log - self.gsw_mu) / self.gsw_sd, (tleaf - self.tleaf_mu) / self.tleaf_sd

    def denormalize(self, gsw_z: np.ndarray, tleaf_z: np.ndarray):
        gsw = np.exp(gsw_z * self.gsw_sd + self.gsw_mu) - GSW_EPS
        tleaf = tleaf_z * self.tleaf_sd + self.tleaf_mu
        return gsw, tleaf

    def to_dict(self) -> dict:
        return {
            "gsw_mu": self.gsw_mu,
            "gsw_sd": self.gsw_sd,
            "tleaf_mu": self.tleaf_mu,
            "tleaf_sd": self.tleaf_sd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TargetScaler":
        s = cls()
        s.gsw_mu, s.gsw_sd = d["gsw_mu"], d["gsw_sd"]
        s.tleaf_mu, s.tleaf_sd = d["tleaf_mu"], d["tleaf_sd"]
        return s
