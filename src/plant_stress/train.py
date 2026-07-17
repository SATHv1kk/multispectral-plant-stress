"""Train the single-frame two-stream FiLM model.

Two-phase schedule:
  1. Warm-up   -- backbone frozen, only the indices tower, gate and head learn.
                  Starting with a trainable backbone would let large random-head
                  gradients wreck the pretrained ImageNet features.
  2. Fine-tune -- upper backbone blocks unfrozen at a lower LR.

Loss is Huber on both heads. The dataset is small and the LI-600 has occasional
wild readings; Huber keeps a single bad label from dominating the gradient the
way squared error would.

Usage:
    python -m plant_stress.train --csv data_with_indices.csv --frames-root OD1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .config import (
    BATCH_SIZE,
    CKPT_DIR,
    EPOCHS_FINETUNE,
    EPOCHS_WARMUP,
    GSW_LOSS_WEIGHT,
    IMG_SIZE,
    LR_FINETUNE,
    LR_WARMUP,
    SEEDS,
    TLEAF_LOSS_WEIGHT,
    WEIGHT_DECAY,
)
from .data.dataset import date_blocked_split, make_single_frame_dataset, resolve_triplets
from .models.two_stream_film import TargetScaler, build_model

# Blocks unfrozen during fine-tuning: the late, most task-specific stages.
FINETUNE_BLOCK_PREFIXES = ("block6", "block7", "top")


def make_optimizer(lr: float, steps_per_epoch: int, epochs: int):
    """AdamW with cosine decay and gradient clipping."""
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        lr, decay_steps=steps_per_epoch * max(1, epochs), alpha=0.1
    )
    return tf.keras.optimizers.AdamW(
        learning_rate=schedule, weight_decay=WEIGHT_DECAY, global_clipnorm=1.0
    )


def set_backbone_trainable(backbone, trainable: bool, only_late_blocks: bool = False) -> None:
    """Freeze or unfreeze the RGB backbone.

    BatchNorm layers are deliberately kept frozen even when unfreezing. With a
    batch size in the low tens, BN would re-estimate its running statistics from
    very noisy batches and drift away from the ImageNet statistics the rest of
    the pretrained filters still assume.
    """
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            continue
        if only_late_blocks:
            layer.trainable = trainable and layer.name.startswith(FINETUNE_BLOCK_PREFIXES)
        else:
            layer.trainable = trainable


def train_one_seed(train_df, val_df, seed: int, out_dir: Path, img_size: int = IMG_SIZE):
    """Train a single seed and return (checkpoint_path, scaler)."""
    tf.keras.utils.set_random_seed(seed)

    scaler = TargetScaler().fit(
        train_df["gsw"].to_numpy("float32"), train_df["Tleaf"].to_numpy("float32")
    )

    train_ds = make_single_frame_dataset(
        train_df, BATCH_SIZE, img_size, training=True, target_scaler=scaler
    )
    val_ds = make_single_frame_dataset(
        val_df, BATCH_SIZE, img_size, training=False, target_scaler=scaler
    )

    model, backbone = build_model(img_size)

    losses = {"gsw": tf.keras.losses.Huber(1.0), "Tleaf": tf.keras.losses.Huber(1.5)}
    loss_weights = {"gsw": GSW_LOSS_WEIGHT, "Tleaf": TLEAF_LOSS_WEIGHT}
    metrics = {"gsw": ["mae"], "Tleaf": ["mae"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"two_stream_film_seed{seed}.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
        ),
    ]

    steps = max(1, math.floor(len(train_df) / BATCH_SIZE))

    print(f"\n=== seed {seed}: warm-up ({EPOCHS_WARMUP} epochs, backbone frozen) ===")
    set_backbone_trainable(backbone, False)
    model.compile(
        optimizer=make_optimizer(LR_WARMUP, steps, EPOCHS_WARMUP),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARMUP, callbacks=callbacks)

    print(f"\n=== seed {seed}: fine-tune ({EPOCHS_FINETUNE} epochs, late blocks) ===")
    set_backbone_trainable(backbone, True, only_late_blocks=True)
    model.compile(  # recompile so the new trainable set takes effect
        optimizer=make_optimizer(LR_FINETUNE, steps, EPOCHS_FINETUNE),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, callbacks=callbacks)

    (out_dir / f"target_scaler_seed{seed}.json").write_text(json.dumps(scaler.to_dict(), indent=2))
    return ckpt_path, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="rows: date,image_file,gsw,Tleaf")
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(CKPT_DIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable float16 compute. Recommended on A100/T4; skip on CPU.",
    )
    args = parser.parse_args()

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    df = pd.read_csv(args.csv)
    df["date"] = df["date"].astype(str)
    df = resolve_triplets(df, args.frames_root)
    df = df.dropna(subset=["gsw", "Tleaf"]).reset_index(drop=True)

    train_df, val_df = date_blocked_split(df)

    for seed in args.seeds:
        path, _ = train_one_seed(train_df, val_df, seed, args.out_dir, args.img_size)
        print(f"[seed {seed}] best checkpoint -> {path}")


if __name__ == "__main__":
    main()
