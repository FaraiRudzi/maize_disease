"""Split a labelled maize-image folder into train/val/test folders.

Works with either layout:
  1. Folders named after the image classes anywhere under --src.
  2. Supervisely format (ann/ and img/ folders), as downloaded from Dataset Ninja.
     Class comes from an image tag or annotated object; unlabelled annotations are skipped.

Usage:
  python prepare_dataset.py --src path/to/downloaded --dst dataset_split
Options: --size 512 (longest side after resizing), --val 0.15, --test 0.15, --seed 42, --max-per-class N (quick tests)
"""
import argparse
import hashlib
import json
import os
import re
import random
from collections import Counter, defaultdict

from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_ALIASES = {
    "healthy": "healthy",
    "health": "healthy",
    "common rust": "common_rust",
    "rust": "common_rust",
    "gray leaf spot": "gray_leaf_spot",
    "grey leaf spot": "gray_leaf_spot",
    "maize leaf blight": "maize_leaf_blight",
    "leaf blight": "maize_leaf_blight",
    "mlb": "maize_leaf_blight",
    "maize streak virus": "maize_streak_virus",
    "streak": "maize_streak_virus",
    "msv": "maize_streak_virus",
    "northern leaf blight": "northern_leaf_blight",
    "nlb": "northern_leaf_blight",
    "magnesium": "magnesium_deficiency",
    "magnesium deficiency": "magnesium_deficiency",
    "magnessium": "magnesium_deficiency",
    "magnessium deficiency": "magnesium_deficiency",
    "nitrogen": "nitrogen_deficiency",
    "nitrogen deficiency": "nitrogen_deficiency",
    "phosphorus": "phosphorus_deficiency",
    "phosphorus deficiency": "phosphorus_deficiency",
    "phosphorous": "phosphorus_deficiency",
    "phosphorous deficiency": "phosphorus_deficiency",
    "potassium": "potassium_deficiency",
    "potassium deficiency": "potassium_deficiency",
    "pottasium": "potassium_deficiency",
    "pottasium deficiency": "potassium_deficiency",
}


def canon(text):
    name = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    name = " ".join(name.split())
    return CLASS_ALIASES.get(name)


def from_supervisely(src):
    items, unlabelled = [], 0
    for root, _, files in os.walk(src):
        if os.path.basename(root) != "ann":
            continue
        img_dir = os.path.join(os.path.dirname(root), "img")
        for fn in files:
            if not fn.endswith(".json"):
                continue
            img_path = os.path.join(img_dir, fn[:-5])
            if not os.path.exists(img_path):
                continue
            with open(os.path.join(root, fn), encoding="utf-8") as f:
                ann = json.load(f)
            label = None
            for t in ann.get("tags", []):
                label = canon(t.get("name", "")) or canon(t.get("value", "")) or label
            if label is None:
                for o in ann.get("objects", []):
                    label = canon(o.get("classTitle", ""))
                    if label:
                        break
            if label is None:
                unlabelled += 1
                continue
            items.append((img_path, label))
    return items, unlabelled


def from_folders(src):
    items, unknown = [], 0
    for root, _, files in os.walk(src):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in IMG_EXT:
                continue
            label = None
            for part in reversed(os.path.relpath(root, src).split(os.sep)):
                label = canon(part)
                if label:
                    break
            if label is None:
                unknown += 1
                continue
            items.append((os.path.join(root, fn), label))
    return items, unknown


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="dataset_split")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-class", type=int, default=0)
    a = ap.parse_args()

    if not os.path.isdir(a.src):
        raise SystemExit(f"Dataset source folder does not exist: {a.src}")
    if a.size < 1:
        raise SystemExit("--size must be a positive integer.")
    if not 0 <= a.val < 1 or not 0 <= a.test < 1 or a.val + a.test >= 1:
        raise SystemExit("--val and --test must be non-negative and sum to less than 1.")

    has_ann = any(os.path.basename(root) == "ann" for root, _, _ in os.walk(a.src))
    if has_ann:
        items, note = from_supervisely(a.src)
        print("Supervisely layout found. Unlabelled annotations skipped:", note)
    else:
        items, note = from_folders(a.src)
        print("Folder layout. Images skipped because no class folder matched:", note)
    if not items:
        raise SystemExit("No labelled images found. Check --src, or send the folder listing to your supervisor or assistant.")

    seen, by_class, dupes = set(), defaultdict(list), 0
    for path, label in items:
        digest = md5(path)
        if digest in seen:
            dupes += 1
            continue
        seen.add(digest)
        by_class[label].append((path, digest))
    print("Exact duplicate files removed:", dupes)

    rng = random.Random(a.seed)
    summary = defaultdict(Counter)
    for label, files in sorted(by_class.items()):
        rng.shuffle(files)
        if a.max_per_class:
            files = files[:a.max_per_class]
        if len(files) < 3:
            print(f"Skipped class {label!r}: needs at least 3 unique images for train/val/test, found {len(files)}.")
            continue
        n_test = max(1, int(len(files) * a.test)) if a.test else 0
        n_val = max(1, int(len(files) * a.val)) if a.val else 0
        while n_test + n_val >= len(files):
            if n_val >= n_test and n_val:
                n_val -= 1
            elif n_test:
                n_test -= 1
            else:
                break
        parts = {"test": files[:n_test], "val": files[n_test:n_test + n_val], "train": files[n_test + n_val:]}
        for split, chunk in parts.items():
            out_dir = os.path.join(a.dst, split, label)
            os.makedirs(out_dir, exist_ok=True)
            for path, digest in chunk:
                try:
                    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                except Exception as e:  # unreadable file
                    print("Skipped unreadable image:", path, e)
                    continue
                img.thumbnail((a.size, a.size))
                img.save(os.path.join(out_dir, digest[:16] + ".jpg"), "JPEG", quality=90)
                summary[split][label] += 1

    if not summary["train"]:
        raise SystemExit("No images were split. Each included class must have at least 3 unique readable images.")

    os.makedirs(a.dst, exist_ok=True)
    print("\nSplit summary (images per class):")
    for split in ("train", "val", "test"):
        print("  {:5s} {}".format(split, dict(summary[split])))
    with open(os.path.join(a.dst, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: dict(v) for k, v in summary.items()}, f, indent=2)


if __name__ == "__main__":
    main()
