"""Train a MobileNetV2 transfer-learning classifier on the prepared maize dataset.

Usage:
  python train_model.py --data dataset_split --out model
Creates in --out: maize_model.keras, class_names.json, metrics.json, confusion_matrix.png, training_curves.png
The saved model contains its own rescaling layer, so the web app passes raw 0-255 pixels.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

L = tf.keras.layers


def load_split(data_dir, split, img, batch, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, split), image_size=(img, img), batch_size=batch,
        label_mode="int", shuffle=shuffle, seed=42)


def build_model(n_classes, img):
    augment = tf.keras.Sequential([
        L.RandomFlip("horizontal_and_vertical"), L.RandomRotation(0.15), L.RandomZoom(0.15),
        L.RandomContrast(0.15), L.RandomBrightness(0.15)], name="augmentation")
    base = tf.keras.applications.MobileNetV2(input_shape=(img, img, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(img, img, 3))
    x = augment(inputs)
    x = L.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def validate_dataset(data_dir):
    required_splits = ("train", "val", "test")
    missing = [
        split for split in required_splits
        if not os.path.isdir(os.path.join(data_dir, split))
    ]
    if missing:
        raise SystemExit(
            "Prepared dataset is missing: {}.\n"
            "Training needs train, val, and test folders under {!r}.\n"
            "If you have downloaded and extracted the dataset, prepare it first:\n"
            "  py prepare_dataset.py --src dataset\\raw --dst {}\n"
            "Then run:\n"
            "  py train_model.py --data {}".format(", ".join(missing), data_dir, data_dir, data_dir)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_split")
    ap.add_argument("--out", default="model")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fine-tune-epochs", type=int, default=5)
    a = ap.parse_args()
    validate_dataset(a.data)
    os.makedirs(a.out, exist_ok=True)

    train = load_split(a.data, "train", a.img, a.batch, True)
    val = load_split(a.data, "val", a.img, a.batch, False)
    test = load_split(a.data, "test", a.img, a.batch, False)
    classes = train.class_names
    print("Classes:", classes)

    counts = [len(os.listdir(os.path.join(a.data, "train", c))) for c in classes]
    total = sum(counts)
    class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    print("Train images per class:", dict(zip(classes, counts)))

    auto = tf.data.AUTOTUNE
    train, val, test = train.prefetch(auto), val.prefetch(auto), test.prefetch(auto)

    model, base = build_model(len(classes), a.img)
    stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train, validation_data=val, epochs=a.epochs, class_weight=class_weight, callbacks=[stop])

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train, validation_data=val, epochs=a.fine_tune_epochs, class_weight=class_weight, callbacks=[stop])

    y_true, y_pred = [], []
    for xb, yb in test:
        y_true += yb.numpy().tolist()
        y_pred += np.argmax(model.predict(xb, verbose=0), axis=1).tolist()
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    model.save(os.path.join(a.out, "maize_model.keras"))
    with open(os.path.join(a.out, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f)
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "train_counts": dict(zip(classes, counts)), "test_report": report,
                   "confusion_matrix": cm.tolist(), "args": vars(a)}, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "confusion_matrix.png"), dpi=150)

    acc = h1.history["accuracy"] + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(acc, label="train")
    ax.plot(vacc, label="validation")
    ax.axvline(len(h1.history["accuracy"]) - 1, linestyle="--", color="grey")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "training_curves.png"), dpi=150)
    print("Saved model and reports in", a.out)


if __name__ == "__main__":
    main()
