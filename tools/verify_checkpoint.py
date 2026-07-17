"""Verify a released checkpoint against the architecture in `temporal_bigru.py`.

This is the tool that established what the released weights actually are. It
reads the weight tensors straight out of the `.keras` archive with h5py, so it
never depends on Keras being able to deserialize the saved config -- which is
exactly what breaks on Keras >= 3.15.

Run it after changing anything in `models/temporal_bigru.py`. If the shape
multiset stops matching, the released weights no longer load and the change is
wrong, however reasonable it looked.

Usage:
    python tools/verify_checkpoint.py models/best_img_imgonly_seed42.keras
"""

from __future__ import annotations

import argparse
import collections
import io
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np

# Reference values, measured from the released seed-42 checkpoint.
EXPECTED_TOTAL_TENSORS = 522
EXPECTED_TOTAL_PARAMS = 13_959_600
EXPECTED_BACKBONE_TENSORS = 497
EXPECTED_BACKBONE_PARAMS = 10_783_535  # EfficientNetB3 (v1), NOT V2-B3


def read_weight_shapes(path: Path, group: str = "layers") -> dict[str, tuple]:
    """Pull {tensor_name: shape} out of a .keras archive without Keras."""
    with zipfile.ZipFile(path) as archive:
        buffer = io.BytesIO(archive.read("model.weights.h5"))
    shapes: dict[str, tuple] = {}
    with h5py.File(buffer, "r") as handle:
        handle[group].visititems(
            lambda name, obj: shapes.__setitem__(name, obj.shape)
            if isinstance(obj, h5py.Dataset)
            else None
        )
    return shapes


def count(shapes: dict[str, tuple]) -> int:
    return sum(int(np.prod(s)) for s in shapes.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Missing checkpoint: {args.checkpoint}")
        print("Download it first:  gh release download v1.0 --dir models/")
        return 2

    real = read_weight_shapes(args.checkpoint)
    backbone = read_weight_shapes(args.checkpoint, "layers/time_distributed")

    print(f"Checkpoint : {args.checkpoint.name}")
    print(f"  tensors  : {len(real)}   params: {count(real):,}")
    print(f"  backbone : {len(backbone)}   params: {count(backbone):,}")

    from plant_stress.models.temporal_bigru import build_model

    ours = build_model(backbone_weights=None)
    print(f"Reconstruction: tensors {len(ours.weights)}   params {ours.count_params():,}")

    checks = {
        "total tensor count": len(real) == len(ours.weights) == EXPECTED_TOTAL_TENSORS,
        "total param count": count(real) == ours.count_params() == EXPECTED_TOTAL_PARAMS,
        "backbone is EfficientNetB3 v1": (
            len(backbone) == EXPECTED_BACKBONE_TENSORS
            and count(backbone) == EXPECTED_BACKBONE_PARAMS
        ),
        "shape multiset matches": (
            collections.Counter(real.values())
            == collections.Counter(tuple(w.shape) for w in ours.weights)
        ),
    }

    print()
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    if not all(checks.values()):
        print("\nReconstruction does NOT match the released weights.")
        return 1

    # The decisive check: the graph actually accepts the trained weights.
    ours.load_weights(args.checkpoint)
    print("  [PASS] load_weights() accepted the trained weights")
    print("\nVerified: temporal_bigru.build_model() matches this checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
