"""Dataset-backed case storage for the maize diagnosis app.

This project does not use a database; it stores diagnosis records in a JSON file so
training, evaluation and the web app all operate with a simple local dataset.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import config


def _case_file_path():
    path = getattr(config, "CASE_LOG_PATH", None) or getattr(config, "SQLITE_PATH", None)
    if path is None:
        path = os.path.join(config.BASE_DIR, "maize_cases.json")
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def _read_cases():
    path = _case_file_path()
    if not os.path.exists(path):
        _write_cases([])
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle) if os.path.getsize(path) else {"cases": []}
    if isinstance(data, list):
        data = {"cases": data}
    if not isinstance(data, dict):
        data = {"cases": []}
    data.setdefault("cases", [])
    return data["cases"]


def _write_cases(cases):
    path = _case_file_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"cases": cases}, handle, indent=2)


def _normalize_case(case):
    case = dict(case)
    case.setdefault("id", 0)
    case.setdefault("created_at", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    case.setdefault("location_text", "")
    case.setdefault("latitude", None)
    case.setdefault("longitude", None)
    case.setdefault("channel", "web")
    case.setdefault("feedback", "")
    case.setdefault("uncertain", False)
    return case


def init_db():
    _read_cases()


def insert_diagnosis(image_file, predicted_class, confidence, uncertain, location_text, latitude, longitude, channel):
    cases = _read_cases()
    case_id = 1 if not cases else max(int(case["id"]) for case in cases if "id" in case) + 1
    case = {
        "id": case_id,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "uncertain": bool(uncertain),
        "location_text": location_text or "",
        "latitude": latitude,
        "longitude": longitude,
        "channel": channel or "web",
        "feedback": "",
        "image_file": image_file,
    }
    cases.append(_normalize_case(case))
    _write_cases(cases)
    return case_id


def set_feedback(case_id, value):
    cases = _read_cases()
    updated = False
    for case in cases:
        if int(case.get("id", 0)) == int(case_id):
            case["feedback"] = value
            updated = True
            break
    if updated:
        _write_cases(cases)
    return updated


def all_cases():
    return [_normalize_case(case) for case in _read_cases()]


def stats():
    cases = all_cases()
    total = len(cases)
    diseased = sum(1 for case in cases if not case.get("uncertain") and case.get("predicted_class") not in {None, "healthy"})
    uncertain = sum(1 for case in cases if case.get("uncertain"))
    has_feedback = [case for case in cases if case.get("feedback") in {"correct", "incorrect", "unsure"}]
    correct = sum(1 for case in has_feedback if case.get("feedback") == "correct")
    accuracy_pct = round((correct / len(has_feedback)) * 100, 1) if has_feedback else 0.0
    accuracy_text = f"{accuracy_pct:.0f}%"

    by_class = []
    for label, count in sorted(Counter(case.get("predicted_class") for case in cases if case.get("predicted_class")).items(), key=lambda item: (-item[1], item[0])):
        by_class.append({"predicted_class": label, "n": count})

    by_location = []
    loc_counts = Counter((case.get("location_text") or "Unknown").strip() or "Unknown" for case in cases)
    for label, count in sorted(loc_counts.items(), key=lambda item: (-item[1], item[0])):
        by_location.append({"loc": label, "n": count})

    day_counts = Counter(case["created_at"][:10] for case in cases)
    daily = [{"day": day, "n": count} for day, count in sorted(day_counts.items())]

    recent = sorted(cases, key=lambda case: case.get("created_at", ""), reverse=True)
    recent = [{
        "id": case.get("id"),
        "created_at": case.get("created_at"),
        "predicted_class": case.get("predicted_class"),
        "confidence": float(case.get("confidence", 0.0)),
        "uncertain": bool(case.get("uncertain")),
        "location_text": case.get("location_text"),
        "feedback": case.get("feedback"),
        "image_file": case.get("image_file"),
    } for case in recent[:10]]

    return {
        "total": total,
        "diseased": diseased,
        "uncertain": uncertain,
        "accuracy_text": accuracy_text,
        "by_class": by_class,
        "by_location": by_location,
        "daily": daily,
        "recent": recent,
    }
