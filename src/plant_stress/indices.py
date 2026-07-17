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

from .config import EPS, NDVI_FOREGROUND_THRESHOLD


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

    indices = tf.concat(
        [ndvi, ndre, zscore_per_image(nir), zscore_per_image(red_edge)], axis=-1
    )

    mask = tf.cast(ndvi > NDVI_FOREGROUND_THRESHOLD, tf.float32)
    return indices, mask


def apply_mask(rgb: tf.Tensor, indices: tf.Tensor, mask: tf.Tensor):
    """Zero out soil and background in both streams before the network sees them.

    Both streams are masked with the SAME mask so the RGB pixel at (i, j) and
    the index vector at (i, j) always describe the same piece of leaf.
    """
    return rgb * mask, indices * mask
