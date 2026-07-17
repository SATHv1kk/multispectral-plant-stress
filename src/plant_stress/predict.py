"""Inference: horizontal-flip TTA and multi-seed ensembling.

Two variance-reduction tricks, both cheap and both applied at the z-scored
level before converting back to physical units:

  TTA       Average the prediction for an image and its mirror. Leaf stress is
            left-right symmetric, so the mirrored view is an equally valid
            observation of the same plant.
  Ensemble  Average across seeds. On a dataset this small, individual runs land
            in noticeably different minima; the mean is consistently steadier
            than any single seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from .models.temporal_bigru import load_checkpoint


def _flip_single_frame(x: dict) -> dict:
    x = dict(x)
    x["rgb"] = tf.image.flip_left_right(x["rgb"])
    x["indices"] = tf.image.flip_left_right(x["indices"])
    return x


def _flip_sequence(x: dict) -> dict:
    x = dict(x)
    # [B,T,H,W,C]: flip the width axis, leaving batch and time untouched.
    x["rgb_seq"] = tf.reverse(x["rgb_seq"], axis=[3])
    x["idx_seq"] = tf.reverse(x["idx_seq"], axis=[3])
    return x


def predict_with_tta(model, dataset: tf.data.Dataset, sequence: bool = False) -> np.ndarray:
    """Average predictions over the original and horizontally-flipped inputs.

    Returns raw model output, still in the network's z-scored target space.
    """
    inputs = dataset.map(lambda x, y: x)
    flipper = _flip_sequence if sequence else _flip_single_frame

    base = model.predict(inputs, verbose=0)
    flipped = model.predict(inputs.map(flipper), verbose=0)

    def stack(p):
        # Single-frame model returns {'gsw','Tleaf'}; temporal returns a tensor.
        if isinstance(p, dict):
            return np.concatenate([p["gsw"], p["Tleaf"]], axis=1)
        return np.asarray(p)

    return 0.5 * (stack(base) + stack(flipped))


def ensemble_predict(
    checkpoint_paths: list[Path],
    dataset: tf.data.Dataset,
    sequence: bool = True,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Mean (or weighted) TTA prediction across seed checkpoints.

    Args:
        weights: optional per-checkpoint weights. Must sum to 1. Defaults to a
            uniform mean, which is what the reported results use -- a weight
            search on a 16-row validation set overfits the split.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoints given.")

    per_seed = []
    for path in checkpoint_paths:
        model = load_checkpoint(str(path))
        per_seed.append(predict_with_tta(model, dataset, sequence=sequence))
        print(f"[predict] {Path(path).name}: done")

    stacked = np.stack(per_seed, axis=0)
    if weights is None:
        return stacked.mean(axis=0)

    w = np.asarray(weights, dtype=np.float32)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {w.sum()}")
    if len(w) != len(checkpoint_paths):
        raise ValueError("weights and checkpoint_paths must be the same length")
    return (stacked * w[:, None, None]).sum(axis=0)


def denormalize_tleaf(z: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """z-scored Tleaf -> degrees C."""
    return z * sd + mu
