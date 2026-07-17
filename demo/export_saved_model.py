"""Convert the demo .keras file into a TF SavedModel directory.

Why this exists: a `.keras` file stores layer *configs* that must be re-parsed
by the loading Keras version. When a deployment target pins an older Python or
Keras than the one that saved the file, that re-parse can fail even though the
weights are perfectly fine. A SavedModel stores a frozen graph instead, so it
replays without consulting Keras layer configs at all.

`app.py` prefers `saved_model/` when it exists and falls back to the `.keras`
file otherwise, so running this is optional locally and useful before deploying.

Usage:
    python export_saved_model.py
    python export_saved_model.py --keras rgb_singleframe_demo.keras --out saved_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG_SIZE = 224


def export(keras_path: Path, out_dir: Path) -> None:
    model = tf.keras.models.load_model(keras_path, compile=False)
    print(f"Loaded {keras_path}  params={model.count_params():,}")

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
    def serving_fn(x):
        return model(x, training=False)

    tf.saved_model.save(model, str(out_dir), signatures={"serving_default": serving_fn})
    print(f"Wrote SavedModel -> {out_dir}")

    # Verify the export reproduces the source model rather than assuming it does.
    probe = np.random.default_rng(0).uniform(0, 255, (2, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    expected = model.predict(probe, verbose=0)

    reloaded = tf.saved_model.load(str(out_dir))
    infer = reloaded.signatures["serving_default"]
    actual = infer(tf.constant(probe))[list(infer.structured_outputs.keys())[0]].numpy()

    max_diff = float(np.abs(expected - actual).max())
    print(f"Verification: max |keras - savedmodel| = {max_diff:.2e}")
    if max_diff > 1e-4:
        raise SystemExit(f"Export mismatch ({max_diff:.2e}) -- refusing to ship a bad SavedModel.")
    print("Export matches the source model.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras", type=Path, default=Path("rgb_singleframe_demo.keras"))
    parser.add_argument("--out", type=Path, default=Path("saved_model"))
    args = parser.parse_args()
    export(args.keras, args.out)


if __name__ == "__main__":
    main()
