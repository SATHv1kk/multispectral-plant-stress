"""tf.data input pipelines for the single-frame and temporal models.

Both pipelines resolve an (date, image_file) row into a synchronised triplet of
RGB / NIR / red-edge frames, compute the spectral index map, mask out
background, and emit the tensors the corresponding model expects.

Splitting is DATE-BLOCKED, never random. Frames captured on the same day share
illumination, canopy state and treatment schedule, so a random row split would
put near-duplicates on both sides and inflate validation scores.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ..config import (
    IMG_SIZE,
    SEQ_LEN,
    STREAM_NIR,
    STREAM_REDEDGE,
    STREAM_RGB,
    VAL_DATE_FRACTION,
)
from ..indices import apply_mask, compute_indices

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------
def find_frame(frames_root: Path, date: str, stream: str, image_file: str) -> str | None:
    """Resolve one frame path, tolerating the naming drift in the raw capture.

    Frame files accumulated inconsistent zero-padding and extensions across
    capture days (`frame_27.png`, `frame_0027.png`, `..._0027.jpg`). Rather
    than rename the archive, we resolve with fallbacks:
      1. exact filename
      2. same name, other extension
      3. any file in the directory carrying the same `frame_<n>` token
    """
    base = Path(frames_root) / date / stream
    if not base.is_dir():
        return None

    exact = base / image_file
    if exact.exists():
        return str(exact)

    stem, ext = exact.stem, exact.suffix.lower()
    alt = base / f"{stem}{'.jpg' if ext == '.png' else '.png'}"
    if alt.exists():
        return str(alt)

    match = re.search(r"(frame[_-]?\d+)", image_file)
    if match:
        token = match.group(1)
        hits = sorted(p for p in base.iterdir() if token in p.name)
        if hits:
            return str(hits[0])
    return None


def resolve_triplets(df: pd.DataFrame, frames_root: Path) -> pd.DataFrame:
    """Keep only rows where all three streams resolved to a real file."""
    rows = []
    for row in df.itertuples(index=False):
        paths = {
            "rgb_path": find_frame(frames_root, row.date, STREAM_RGB, row.image_file),
            "nir_path": find_frame(frames_root, row.date, STREAM_NIR, row.image_file),
            "re_path": find_frame(frames_root, row.date, STREAM_REDEDGE, row.image_file),
        }
        if all(paths.values()):
            rows.append({**row._asdict(), **paths})

    out = pd.DataFrame(rows)
    dropped = len(df) - len(out)
    if dropped:
        print(f"[dataset] dropped {dropped}/{len(df)} rows with missing frames")
    return out


def date_blocked_split(df: pd.DataFrame, val_fraction: float = VAL_DATE_FRACTION):
    """Split by capture date, holding out the LAST dates for validation.

    Holding out the *latest* dates rather than random dates also makes the
    validation score a forward-in-time estimate, which is the way the model
    would actually be used.
    """
    dates = sorted(df["date"].astype(str).unique())
    cut = int(len(dates) * (1.0 - val_fraction))
    train_dates, val_dates = set(dates[:cut]), set(dates[cut:])
    train = df[df["date"].astype(str).isin(train_dates)].reset_index(drop=True)
    val = df[df["date"].astype(str).isin(val_dates)].reset_index(drop=True)
    print(f"[dataset] dates: {len(train_dates)} train / {len(val_dates)} val")
    print(f"[dataset] rows : {len(train)} train / {len(val)} val")
    return train, val


# --------------------------------------------------------------------------
# Frame loading
# --------------------------------------------------------------------------
def _decode(path: tf.Tensor, channels: int, img_size: int) -> tf.Tensor:
    img = tf.io.decode_image(tf.io.read_file(path), channels=channels, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0, 1]
    return tf.image.resize(img, (img_size, img_size), antialias=True)


def load_and_prepare(rgb_path, nir_path, re_path, img_size: int):
    """Load one triplet and return masked (rgb, indices). No augmentation.

    Augmentation is applied by the caller, so that a clip can be augmented as a
    unit -- see `augment`.
    """
    rgb = _decode(rgb_path, 3, img_size)
    nir = _decode(nir_path, 1, img_size)
    red_edge = _decode(re_path, 1, img_size)

    idx, mask = compute_indices(rgb, nir, red_edge)
    return apply_mask(rgb, idx, mask)


def augment(rgb: tf.Tensor, idx: tf.Tensor):
    """Augment a single frame [H,W,C] or a whole clip [T,H,W,C].

    `tf.image` ops treat a leading axis as batch, so passing a [T,H,W,C] clip
    applies ONE flip decision to every frame in it. That matters: flipping some
    frames of a clip and not others would destroy the temporal continuity the
    BiGRU exists to read.

    RGB and indices are flipped with the same decision so they stay
    pixel-aligned. Photometric jitter touches RGB only -- perturbing the NIR or
    red-edge bands would corrupt the physics baked into NDVI/NDRE.
    """
    if tf.random.uniform([]) < 0.5:
        rgb = tf.image.flip_left_right(rgb)
        idx = tf.image.flip_left_right(idx)

    rgb = tf.image.random_brightness(rgb, 0.05)
    rgb = tf.image.random_contrast(rgb, 0.9, 1.1)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    return rgb, idx


def make_single_frame_dataset(
    df: pd.DataFrame,
    batch_size: int,
    img_size: int = IMG_SIZE,
    training: bool = False,
    target_scaler=None,
) -> tf.data.Dataset:
    """Dataset for `models.two_stream_film`: {'rgb','indices'} -> {'gsw','Tleaf'}.

    Args:
        target_scaler: a fitted `two_stream_film.TargetScaler`. Targets are
            emitted in the network's normalised space when provided, and in raw
            physical units when None (useful for evaluation).
    """
    gsw = df["gsw"].to_numpy(np.float32)
    tleaf = df["Tleaf"].to_numpy(np.float32)
    if target_scaler is not None:
        gsw, tleaf = target_scaler.normalize(gsw, tleaf)
        gsw, tleaf = gsw.astype(np.float32), tleaf.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices(
        (
            df["rgb_path"].tolist(),
            df["nir_path"].tolist(),
            df["re_path"].tolist(),
            gsw,
            tleaf,
        )
    )
    if training:
        ds = ds.shuffle(max(2, len(df)), reshuffle_each_iteration=True)

    def _map(rgb_p, nir_p, re_p, g, t):
        rgb, idx = load_and_prepare(rgb_p, nir_p, re_p, img_size)
        if training:
            rgb, idx = augment(rgb, idx)
        return {"rgb": rgb, "indices": idx}, {"gsw": g, "Tleaf": t}

    return (
        ds.map(_map, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=training)
        .prefetch(AUTOTUNE)
    )


def make_sequence_dataset(
    clips: list[dict],
    batch_size: int,
    seq_len: int = SEQ_LEN,
    img_size: int = IMG_SIZE,
    training: bool = False,
) -> tf.data.Dataset:
    """Dataset for `models.temporal_bigru`: {'rgb_seq','idx_seq'} -> Tleaf.

    Args:
        clips: one dict per clip with keys 'rgb_paths', 'nir_paths', 're_paths'
            (each a list of `seq_len` paths, in time order) and 'tleaf'.
    """
    rgb = tf.constant([c["rgb_paths"] for c in clips])   # [N, T]
    nir = tf.constant([c["nir_paths"] for c in clips])
    red = tf.constant([c["re_paths"] for c in clips])
    y = tf.constant([c["tleaf"] for c in clips], dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((rgb, nir, red, y))
    if training:
        ds = ds.shuffle(max(2, len(clips)), reshuffle_each_iteration=True)

    def _map(rgb_paths, nir_paths, re_paths, tleaf):
        frames = [
            load_and_prepare(rgb_paths[i], nir_paths[i], re_paths[i], img_size)
            for i in range(seq_len)
        ]
        rgb_seq = tf.stack([f[0] for f in frames], axis=0)  # [T,H,W,3]
        idx_seq = tf.stack([f[1] for f in frames], axis=0)  # [T,H,W,4]
        if training:
            # Augment the stacked clip so one flip decision covers all T frames.
            rgb_seq, idx_seq = augment(rgb_seq, idx_seq)
        return {"rgb_seq": rgb_seq, "idx_seq": idx_seq}, tleaf

    return (
        ds.map(_map, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=training)
        .prefetch(AUTOTUNE)
    )
