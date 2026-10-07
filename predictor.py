"""Loads the trained model once and classifies a leaf photo.

TensorFlow is imported lazily so the web app can start (and be tested) before a model exists.
The saved model contains its own preprocessing, so raw 0-255 pixel values are passed in.
"""
import json
import os

import numpy as np
from PIL import Image

import config

_model = None
_classes = None


class ModelNotReady(Exception):
    """Raised when no trained model has been found."""


def is_ready():
    return os.path.exists(config.MODEL_PATH) and os.path.exists(config.CLASSES_PATH)


def _load():
    global _model, _classes
    if _model is not None:
        return
    if not is_ready():
        raise ModelNotReady()
    import tensorflow as tf
    _model = tf.keras.models.load_model(config.MODEL_PATH)
    with open(config.CLASSES_PATH, encoding="utf-8") as f:
        _classes = json.load(f)


def predict(image_path):
    _load()
    img = Image.open(image_path).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    batch = np.expand_dims(np.asarray(img, dtype="float32"), 0)
    probs = _model.predict(batch, verbose=0)[0]
    best = int(np.argmax(probs))
    confidence = float(probs[best])
    ranking = sorted(zip(_classes, (float(p) for p in probs)), key=lambda t: -t[1])
    return {
        "label": _classes[best],
        "confidence": confidence,
        "scores": ranking,
        "uncertain": confidence < config.CONFIDENCE_THRESHOLD,
    }
