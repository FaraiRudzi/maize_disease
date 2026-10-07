"""Maize Leaf Check: Flask web application."""
import csv
import hmac
import io
import json
import os
import uuid
from functools import wraps

from flask import (Flask, Response, abort, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from PIL import Image, ImageOps, UnidentifiedImageError

import config
import db
import predictor

app = Flask(__name__)
app.config.update(SECRET_KEY=config.SECRET_KEY, MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024)
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
db.init_db()

with open(os.path.join(config.BASE_DIR, "advice.json"), encoding="utf-8") as _f:
    ADVICE = json.load(_f)

LABEL_NAMES = {"healthy": "Healthy leaf", "maize_leaf_blight": "Maize leaf blight",
               "maize_streak_virus": "Maize streak virus"}


def label_name(label):
    return LABEL_NAMES.get(label, label.replace("_", " ").capitalize())


# ---------- helpers ----------
def save_upload(storage):
    if storage is None or not storage.filename:
        raise ValueError("Choose a photo of a maize leaf first.")
    ext = storage.filename.rsplit(".", 1)[-1].lower() if "." in storage.filename else ""
    if ext not in config.ALLOWED_EXT:
        raise ValueError("Upload a JPG, PNG or WEBP photo.")
    try:
        img = Image.open(storage.stream)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("That file could not be read as a photo. Try another one.")
    img = ImageOps.exif_transpose(img).convert("RGB")  # re-encoding also strips GPS data stored in the photo
    img.thumbnail((1280, 1280))
    name = uuid.uuid4().hex + ".jpg"
    img.save(os.path.join(config.UPLOAD_DIR, name), "JPEG", quality=90)
    return name


def parse_location(form):
    text = (form.get("location") or "").strip()[:120].title()
    def num(key, lo, hi):
        try:
            v = float(form.get(key, ""))
        except ValueError:
            return None
        return round(v, 2) if lo <= v <= hi else None
    return text, num("lat", -90, 90), num("lon", -180, 180)


def run_diagnosis(storage, form, channel="web"):
    name = save_upload(storage)
    path = os.path.join(config.UPLOAD_DIR, name)
    try:
        pred = predictor.predict(path)
    except predictor.ModelNotReady:
        os.remove(path)
        raise
    text, lat, lon = parse_location(form)
    case_id = db.insert_diagnosis(name, pred["label"], pred["confidence"], pred["uncertain"], text, lat, lon, channel)
    advice = ADVICE["uncertain"] if pred["uncertain"] else ADVICE.get(pred["label"], ADVICE["uncertain"])
    return {
        "case_id": case_id,
        "label": pred["label"],
        "label_name": label_name(pred["label"]),
        "confidence": round(pred["confidence"] * 100, 1),
        "uncertain": pred["uncertain"],
        "scores": [{"label": l, "name": label_name(l), "pct": round(p * 100, 1)} for l, p in pred["scores"]],
        "advice": advice,
    }


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if config.DASHBOARD_PASSWORD and not session.get("officer"):
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper


NOT_READY = "The model has not been trained yet. Train it first (see README, step 3)."


# ---------- farmer-facing pages ----------
@app.get("/")
def index():
    return render_template("index.html", thanks=request.args.get("thanks"))


@app.post("/diagnose")
def diagnose():
    try:
        result = run_diagnosis(request.files.get("photo"), request.form)
    except ValueError as e:
        return render_template("index.html", error=str(e)), 400
    except predictor.ModelNotReady:
        return render_template("index.html", error=NOT_READY), 503
    return render_template("index.html", result=result)


@app.post("/feedback/<int:case_id>")
def feedback(case_id):
    value = request.form.get("value")
    if value in {"correct", "incorrect", "unsure"}:
        db.set_feedback(case_id, value)
    return redirect(url_for("index", thanks=1))


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error="That photo is too large. Maximum size is {} MB.".format(config.MAX_UPLOAD_MB)), 413


# ---------- JSON API (for future chat or mobile channels) ----------
@app.post("/api/predict")
def api_predict():
    try:
        result = run_diagnosis(request.files.get("photo"), request.form, channel="api")
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except predictor.ModelNotReady:
        return jsonify(error=NOT_READY), 503
    return jsonify(result)


@app.get("/health")
def health():
    return jsonify(status="ok", model_ready=predictor.is_ready())


# ---------- extension officer area ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if config.DASHBOARD_PASSWORD and hmac.compare_digest(request.form.get("password", ""), config.DASHBOARD_PASSWORD):
            session["officer"] = True
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
        error = "Wrong password."
    return render_template("login.html", error=error)


@app.get("/logout")
def logout():
    session.pop("officer", None)
    return redirect(url_for("index"))


@app.get("/dashboard")
@login_required
def dashboard():
    s = db.stats()
    for r in s["by_class"]:
        r["name"] = label_name(r["predicted_class"])
    for r in s["recent"]:
        r["name"] = "Uncertain" if r["uncertain"] else label_name(r["predicted_class"])
        r["pct"] = round(r["confidence"] * 100)
    return render_template("dashboard.html", s=s, protected=bool(config.DASHBOARD_PASSWORD))


@app.get("/export.csv")
@login_required
def export_csv():
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["id", "created_at", "predicted_class", "confidence", "uncertain", "location", "latitude",
                "longitude", "channel", "feedback", "image_file"])
    for r in db.all_cases():
        w.writerow([r["id"], r["created_at"], r["predicted_class"], round(r["confidence"], 4), r["uncertain"],
                    r["location_text"] or "", r["latitude"] if r["latitude"] is not None else "",
                    r["longitude"] if r["longitude"] is not None else "", r["channel"], r["feedback"] or "", r["image_file"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=maize_cases.csv"})


@app.get("/uploads/<path:name>")
@login_required
def uploaded(name):
    if "/" in name or ".." in name:
        abort(404)
    return send_from_directory(config.UPLOAD_DIR, name)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
