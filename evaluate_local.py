"""Test the trained model on your own Zimbabwean maize photos.

Put photos in folders named after the classes:
  local_test/healthy/*.jpg   local_test/maize_leaf_blight/*.jpg   local_test/maize_streak_virus/*.jpg
Label them with help from an extension officer or agriculture lecturer, then run:
  python evaluate_local.py --folder local_test
Writes model/local_eval.json (use the numbers in your results chapter).
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="local_test")
    a = ap.parse_args()
    model = tf.keras.models.load_model(config.MODEL_PATH)
    with open(config.CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    y_true, y_pred = [], []
    for ci, cname in enumerate(classes):
        d = os.path.join(a.folder, cname)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if os.path.splitext(fn)[1].lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            img = Image.open(os.path.join(d, fn)).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
            p = model.predict(np.expand_dims(np.asarray(img, dtype="float32"), 0), verbose=0)[0]
            y_true.append(ci)
            y_pred.append(int(np.argmax(p)))
    if not y_true:
        raise SystemExit("No photos found in " + a.folder)
    print(classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes, zero_division=0))
    out = {"n_images": len(y_true),
           "report": classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes,
                                           output_dict=True, zero_division=0),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist()}
    path = os.path.join(os.path.dirname(config.MODEL_PATH), "local_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved", path)


if __name__ == "__main__":
    main()
