"""Gradio demo: predict leaf stress indicators from a single RGB photo.

This is the lightweight demo model, not the research model:
  * single frame, RGB only (EfficientNetV2-B0, 224x224, ~6.1M params)
  * no NIR / red-edge, no NDVI / NDRE, no FiLM fusion, no temporal context

The research models live in `src/plant_stress/models/`. Predictions here are raw
and illustrative, not agronomic-grade.

Runs unchanged on a Hugging Face ZeroGPU Space and on a local machine:
    python app.py
"""

import os

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------- Config
IMG_SIZE = 224

# Target order used during training: ['gsw', 'Tleaf'].
GSW_IDX, TLEAF_IDX = 0, 1

# Prefer the TF SavedModel when present. It replays a frozen graph instead of
# re-parsing Keras layer configs, so it loads regardless of the Keras version
# the Space happens to pin. Fall back to the .keras file, which is what ships in
# the git repo.
SAVED_MODEL_DIR = "saved_model"
KERAS_PATH = "rgb_singleframe_demo.keras"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# `spaces` only exists on Hugging Face ZeroGPU hardware. Import it when it is
# available so the Space still gets its GPU allocation; otherwise fall back to a
# no-op decorator so this same file runs locally.
try:
    import spaces

    gpu_decorator = spaces.GPU
except ImportError:

    def gpu_decorator(fn):
        return fn


# ------------------------------------------------------- Load model once
def _load_model():
    """Return (predict_fn, description). Tries SavedModel, then .keras."""
    if os.path.isdir(SAVED_MODEL_DIR):
        loaded = tf.saved_model.load(SAVED_MODEL_DIR)
        infer = loaded.signatures["serving_default"]
        out_key = list(infer.structured_outputs.keys())[0]

        def predict_fn(batch):
            return infer(tf.constant(batch))[out_key].numpy()

        return predict_fn, f"TF SavedModel ({SAVED_MODEL_DIR})"

    model = tf.keras.models.load_model(KERAS_PATH, compile=False)

    def predict_fn(batch):
        return model.predict(batch, verbose=0)

    return predict_fn, f"Keras model ({KERAS_PATH})"


print("Loading model...")
_predict, _source = _load_model()
print("Loaded:", _source)


# ------------------------------------------- Stress thresholds (thesis)
def drought_level(gsw: float) -> str:
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
    if tleaf < 25:
        return "Normal"
    if tleaf < 28:
        return "Low"
    if tleaf < 30:
        return "Medium"
    if tleaf < 35:
        return "High"
    return "Severe"


# --------------------------------------------------------- Preprocessing
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """Resize to the training resolution and hand over pixels in [0, 255].

    `efficientnet_v2.preprocess_input` is a documented pass-through: the
    EfficientNetV2 graph carries its own Rescaling + Normalization layers, so
    normalising here as well would scale the input twice.
    """
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype="float32")


@gpu_decorator
def predict_raw(pil_img: Image.Image):
    """Run the model with horizontal-flip TTA -> raw (gsw, Tleaf)."""
    base = preprocess(pil_img)
    flipped = base[:, ::-1, :]

    batch = np.stack([base, flipped], axis=0).astype("float32")
    values = np.asarray(_predict(batch))
    averaged = values.mean(axis=0)
    return float(averaged[GSW_IDX]), float(averaged[TLEAF_IDX])


def predict(image: Image.Image):
    if image is None:
        return "Upload a leaf or crop photo to run a prediction."

    gsw_raw, tleaf_raw = predict_raw(image)

    # The head is an unconstrained linear layer, so gsw can come out negative on
    # an out-of-distribution image. Surface that rather than hide it: a negative
    # stomatal conductance is physically impossible and is the clearest signal
    # that the photo is nothing like the training data.
    note = ""
    if gsw_raw < 0:
        note = (
            "\n\n> **Note:** predicted gsw is negative, which is physically "
            "impossible. That usually means the image sits far outside the "
            "training distribution (oat/barley leaves on the Maynooth farmbed)."
        )

    return f"""
### Predicted physiological stress indicators

| Metric | Predicted value | Stress level |
|---|---|---|
| Leaf temperature (Tleaf) | **{tleaf_raw:.2f} °C** | {heat_level(tleaf_raw)} |
| Stomatal conductance (gsw) | {gsw_raw:.4f} mol·m⁻²·s⁻¹ | {drought_level(gsw_raw)} (indicative only) |

---
**Model:** single-frame RGB EfficientNetV2-B0 with horizontal-flip TTA.

Tleaf is the reliable output. gsw is shown for transparency and is **not** a
validated result — it is physiologically volatile and hard to infer from one
static frame.

Research prototype trained on a small farmbed dataset. Illustrative, not
agronomic-grade.{note}
"""


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a leaf or crop photo"),
    outputs=gr.Markdown(label="Prediction"),
    title="Plant Physiological Stress Predictor",
    description=(
        "Upload a photo of an oat/barley leaf. Predictions (Tleaf and gsw) come "
        "from a single-frame RGB model and are raw model output — research-grade, "
        "not agronomic-grade measurements. Full project: https://sathvik.info"
    ),
    article=(
        "The full research model fuses NDVI/NDRE spectral indices through a "
        "two-stream FiLM architecture; this demo runs the RGB path only. "
        "Code: https://github.com/Honeybadzer0007/multispectral-plant-stress"
    ),
)

if __name__ == "__main__":
    demo.launch()
