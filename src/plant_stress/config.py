"""Central configuration: paths, image geometry, and stress thresholds.

Every magic number used across training, inference and the demo lives here so
the thesis, the repo and the live demo cannot drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# The project was developed in Google Colab against a Drive mount. Override any
# of these with environment variables when running elsewhere.
DRIVE_ROOT = Path(os.environ.get("PLANT_ROOT", "/content/drive/MyDrive/PLANT"))

CSV_MATCHED = DRIVE_ROOT / "licor_matched_frame" / "data_with_indices.csv"
CSV_TRAIN = DRIVE_ROOT / "licor_matched_frame" / "train_data.csv"
CSV_VAL = DRIVE_ROOT / "licor_matched_frame" / "validation_data.csv"

# Frames live under FRAMES_ROOT/<date>/<stream>/, where stream is one of:
#   rgb   -> RGB camera
#   left  -> near-infrared (NIR)
#   right -> red-edge
FRAMES_ROOT = DRIVE_ROOT / "OD1"
STREAM_RGB, STREAM_NIR, STREAM_REDEDGE = "rgb", "left", "right"

CKPT_DIR = DRIVE_ROOT / "model_ckpts_img"
PRED_DIR = DRIVE_ROOT / "preds"

# --------------------------------------------------------------------------
# Image / sequence geometry
# --------------------------------------------------------------------------
IMG_SIZE = 320          # research model (two-stream and temporal)
DEMO_IMG_SIZE = 224     # lightweight single-frame RGB demo model
SEQ_LEN = 5             # frames per clip for the temporal model
INDEX_CHANNELS = 4      # NDVI, NDRE, z(NIR), z(red-edge)

# NDVI value above which a pixel is treated as foreground vegetation.
# Everything below is suppressed so the network never sees soil/background.
NDVI_FOREGROUND_THRESHOLD = 0.15

EPS = 1e-6              # division guard for index maths
GSW_EPS = 1e-4          # offset for log(gsw + eps); see notes in train.py

# ImageNet statistics, used when the backbone is built WITHOUT its own
# preprocessing layers. NOTE: the shipped checkpoints were trained with
# `include_preprocessing=True`, i.e. the backbone normalises internally.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
SEEDS = (42, 123, 777)
BATCH_SIZE = 24
EPOCHS_WARMUP = 4
EPOCHS_FINETUNE = 70
LR_WARMUP = 3e-4
LR_FINETUNE = 8e-5
WEIGHT_DECAY = 1e-4
GSW_LOSS_WEIGHT = 1.5   # gsw is the harder target, so it is up-weighted
TLEAF_LOSS_WEIGHT = 1.0

# Validation uses the LAST 25% of capture *dates*, never a random row split.
# A random split would leak: frames from one date are highly correlated.
VAL_DATE_FRACTION = 0.25

# --------------------------------------------------------------------------
# Stress thresholds
# --------------------------------------------------------------------------
# These are PROJECT-SPECIFIC interpretive bands chosen for this Maynooth
# farmbed trial on oats and barley. They are NOT universal agronomic standards.

STRESS_ORDER = ("Normal", "Low", "Medium", "High", "Severe")


def drought_level(gsw: float) -> str:
    """Map stomatal conductance (mol m-2 s-1) to a drought stress band.

    Lower gsw means more closed stomata, i.e. more drought stress.
    """
    if gsw < 0.05:
        return "Severe"
    if gsw < 0.10:
        return "High"
    if gsw < 0.20:
        return "Medium"
    if gsw <= 0.30:
        return "Low"
    return "Normal"


def heat_level(tleaf: float) -> str:
    """Map leaf temperature (deg C) to a heat stress band.

    Higher Tleaf means the leaf is shedding less heat via transpiration.
    """
    if tleaf < 25.0:
        return "Normal"
    if tleaf < 28.0:
        return "Low"
    if tleaf < 30.0:
        return "Medium"
    if tleaf < 35.0:
        return "High"
    return "Severe"
