"""Per-pixel spectral indices and the vegetation foreground mask.

The gantry rig records three synchronised streams: RGB, near-infrared (NIR) and
red-edge. From those we build a 4-channel index map that the spectral stream of
the network consumes:

    channel 0  NDVI  = (NIR - Red)      / (NIR + Red)
    channel 1  NDRE  = (NIR - RedEdge)  / (NIR + RedEdge)
    channel 2  z(NIR)       per-image z-score
    channel 3  z(RedEdge)   per-image z-score

NDVI keys on chlorophyll absorption in the red band, so it separates vegetation
from soil. NDRE uses the red-edge band instead, which saturates far later than
NDVI in dense canopy and therefore still varies once NDVI has flattened out.

The NIR and red-edge bands are z-scored *per image* rather than globally: the
rig has no radiometric calibration target, so absolute band values drift with
illumination between capture days. Per-image standardisation removes that drift
and keeps the network from learning "which day is this" instead of "how
stressed is this plant".
"""

from __future__ import annotations

import tensorflow as tf

from .config import EPS, IMAGENET_MEAN, IMAGENET_STD, NDVI_FOREGROUND_THRESHOLD


def normalized_difference(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Normalised difference (a - b) / (a + b), guarded against divide-by-zero."""
    return (a - b) / (a + b + EPS)


def zscore_per_image(x: tf.Tensor) -> tf.Tensor:
    """Standardise a single image band to zero mean, unit variance."""
    mean = tf.reduce_mean(x)
    std = tf.math.reduce_std(x) + EPS
    return (x - mean) / std


def compute_indices(rgb: tf.Tensor, nir: tf.Tensor, red_edge: tf.Tensor):
    """Build the 4-channel index map and the vegetation mask for one frame.

    Args:
        rgb: float tensor [H, W, 3] in [0, 1], channels in R, G, B order.
        nir: float tensor [H, W, 1] in [0, 1].
        red_edge: float tensor [H, W, 1] in [0, 1].

    Returns:
        indices: float tensor [H, W, 4] -> NDVI, NDRE, z(NIR), z(red-edge).
        mask: float tensor [H, W, 1], 1.0 on vegetation and 0.0 elsewhere.
    """
    red = rgb[..., 0:1]

    ndvi = normalized_difference(nir, red)
    ndre = normalized_difference(nir, red_edge)

    # Clip before stacking, matching the original training pipeline.
    # NDVI/NDRE are mathematically bounded to [-1, 1], but the epsilon guard and
    # float error let them drift marginally outside; clipping pins the range.
    # The z-scored bands are clipped to +/-3 sigma so a handful of specular
    # highlights cannot hand the network an input two orders of magnitude larger
    # than everything else.
    indices = tf.concat(
        [
            tf.clip_by_value(ndvi, -1.0, 1.0),
            tf.clip_by_value(ndre, -1.0, 1.0),
            tf.clip_by_value(zscore_per_image(nir), -3.0, 3.0),
            tf.clip_by_value(zscore_per_image(red_edge), -3.0, 3.0),
        ],
        axis=-1,
    )

    # Mask is derived from UNCLIPPED ndvi -- identical either way, since the
    # 0.15 threshold sits well inside the clip bounds.
    mask = tf.cast(ndvi > NDVI_FOREGROUND_THRESHOLD, tf.float32)
    return indices, mask


def normalize_rgb(rgb: tf.Tensor) -> tf.Tensor:
    """ImageNet-normalise an RGB tensor already scaled to [0, 1].

    This is applied EXPLICITLY in the pipeline rather than left to the
    backbone's built-in preprocessing, for two reasons:

    1. It matches the original training pipeline, which normalised here.
    2. It lets masking happen AFTER normalisation, so masked background is
       exactly 0.0. If the backbone normalised internally instead, masked
       pixels would arrive as 0 and be mapped to roughly -2.1 -- i.e. the
       background would read as a saturated *black leaf* rather than as
       "nothing here".

    The models that consume this MUST be built with preprocessing disabled
    (`include_preprocessing=False`), or the input is normalised twice.
    """
    mean = tf.constant(IMAGENET_MEAN, tf.float32)
    std = tf.constant(IMAGENET_STD, tf.float32)
    return (rgb - mean) / std


def apply_mask(rgb: tf.Tensor, indices: tf.Tensor, mask: tf.Tensor):
    """Zero out soil and background in both streams.

    Expects `rgb` to be ALREADY normalised (see `normalize_rgb`), so masked
    pixels come out as exact zeros.

    Both streams are masked with the SAME mask so the RGB pixel at (i, j) and
    the index vector at (i, j) always describe the same piece of leaf.
    """
    return rgb * mask, indices * mask
