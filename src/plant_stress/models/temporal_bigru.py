"""Temporal two-stream network with a bidirectional GRU. Predicts Tleaf only.

THIS is the architecture of the released checkpoints
(`best_img_imgonly_seed{42,123,777}.keras`). It is NOT the single-frame FiLM
model in `two_stream_film.py`; see docs/architecture.md and models/README.md
for why both exist.

    rgb_seq [T,320,320,3] --TimeDistributed(EfficientNet-B3, avg pool)--> [T,1536]
    idx_seq [T,320,320,4] --TimeDistributed(indices CNN)---------------> [T,  96]
                                     concat                           -> [T,1632]
                                     Dropout(0.2)
                                     Bidirectional(GRU(256))           -> [512]
                                     Dense(384, gelu)
                                     Dropout(0.3)
                                     Dense(1) -> Tleaf (z-scored)

IMPORTANT -- backbone identity. The RGB encoder here is `EfficientNetB3`, the
ORIGINAL v1 EfficientNet, NOT `EfficientNetV2B3`. This was verified against the
released weights: the checkpoint's backbone has 497 weight tensors and
10,783,535 parameters, an exact shape-for-shape match with EfficientNetB3 and a
mismatch with EfficientNetV2B3 (536 tensors / 12,930,622 params). The
checkpoint also contains `block7a`/`block7b` and depthwise `block1a_dwconv`,
neither of which exists in EfficientNetV2-B3 (V2 has no block7 and uses fused
convolutions in its early blocks).

The thesis and the single-frame model use EfficientNetV2-B3. These temporal
checkpoints do not. Do not "fix" this to V2 -- it would silently stop the
released weights from loading.

Two deliberate differences from the single-frame model:

1. Fusion is plain concatenation, not FiLM gating. Once a BiGRU sits downstream
   it can learn cross-stream interactions over time itself, so the gate was
   redundant here.
2. There is only a Tleaf head. gsw proved too volatile to regress reliably from
   this dataset (stomata respond on a timescale of minutes), so no gsw output is
   claimed. This mirrors the honest reporting in the thesis.

The backbone carries its own Rescaling + Normalization layers, so feed it RGB in
[0, 1] and do NOT apply ImageNet normalisation yourself.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import models as M

from ..config import IMG_SIZE, INDEX_CHANNELS, SEQ_LEN

RGB_FEATURES = 1536   # EfficientNet-B3 pooled feature width
IDX_FEATURES = 96
GRU_UNITS = 256       # bidirectional -> 512 out
HEAD_UNITS = 384


def build_rgb_encoder(img_size: int = IMG_SIZE, weights: str | None = "imagenet") -> M.Model:
    """EfficientNet-B3 (v1), average-pooled to a 1536-d vector per frame.

    See the module docstring: this is v1, verified against the released weights.
    """
    return tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
        pooling="avg",
    )


def build_indices_encoder(img_size: int = IMG_SIZE) -> M.Model:  # noqa: D401
    """Compact strided CNN over the index map, pooled to a 96-d vector."""
    inp = L.Input(shape=(img_size, img_size, INDEX_CHANNELS))
    x = inp
    for filters in (32, 64, IDX_FEATURES):
        x = L.Conv2D(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("gelu")(x)
    x = L.GlobalAveragePooling2D()(x)
    return M.Model(inp, x, name="idx_cnn")


def build_model(
    seq_len: int = SEQ_LEN,
    img_size: int = IMG_SIZE,
    backbone_weights: str | None = "imagenet",
) -> M.Model:
    """Build the temporal Tleaf regressor matching the released checkpoints.

    Args:
        backbone_weights: 'imagenet' to start from pretrained weights when
            training. Pass None when you are about to restore a checkpoint
            anyway -- it skips a 44 MB download that would just be overwritten.
    """
    rgb_in = L.Input(shape=(seq_len, img_size, img_size, 3), name="rgb_seq")
    idx_in = L.Input(shape=(seq_len, img_size, img_size, INDEX_CHANNELS), name="idx_seq")

    rgb_encoder = build_rgb_encoder(img_size, weights=backbone_weights)
    rgb_encoder._name = "effb3_backbone"
    idx_encoder = build_indices_encoder(img_size)

    r_seq = L.TimeDistributed(rgb_encoder, name="td_rgb")(rgb_in)
    s_seq = L.TimeDistributed(idx_encoder, name="td_idx")(idx_in)

    x = L.Concatenate(axis=-1, name="feat_cat")([r_seq, s_seq])
    x = L.Dropout(0.2)(x)
    x = L.Bidirectional(L.GRU(GRU_UNITS, return_sequences=False), name="bigru")(x)
    x = L.Dense(HEAD_UNITS, activation="gelu")(x)
    x = L.Dropout(0.3)(x)
    tleaf = L.Dense(1, name="tleaf")(x)

    return M.Model(
        inputs={"rgb_seq": rgb_in, "idx_seq": idx_in},
        outputs=tleaf,
        name="temporal_bigru_tleaf",
    )


def load_checkpoint(path: str, seq_len: int = SEQ_LEN, img_size: int = IMG_SIZE) -> M.Model:
    """Load a released checkpoint for inference, robust across Keras versions.

    The checkpoints were written by Keras 3.10. On Keras >= 3.15,
    `load_model()` raises `ValueError: Cannot convert '5' to a shape` while
    deserializing the saved `TimeDistributed` config -- a Keras regression, not
    a problem with the weights themselves.

    So we try the direct load first, then fall back to rebuilding the graph here
    and loading only the weight tensors, which skips config deserialization
    entirely. The fallback is verified: it reproduces the checkpoint's 522
    weight tensors / 13,959,600 parameters exactly.
    """
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        model = build_model(seq_len=seq_len, img_size=img_size, backbone_weights=None)
        model.load_weights(path)
        return model
