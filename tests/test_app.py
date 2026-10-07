import io
import os
import sys
import tempfile
import unittest
from unittest import mock

_tmp = tempfile.mkdtemp()
os.environ["SQLITE_PATH"] = os.path.join(_tmp, "test.db")
os.environ["UPLOAD_DIR"] = os.path.join(_tmp, "uploads")
os.environ["DASHBOARD_PASSWORD"] = "secret"
os.environ["MODEL_PATH"] = os.path.join(_tmp, "none.keras")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import app as app_module  # noqa: E402
import predictor  # noqa: E402


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (40, 120, 50)).save(buf, "JPEG")
    buf.seek(0)
    return buf


FAKE = {"label": "maize_streak_virus", "confidence": 0.93, "uncertain": False,
        "scores": [("maize_streak_virus", 0.93), ("healthy", 0.05), ("maize_leaf_blight", 0.02)]}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.c = app_module.app.test_client()

    def test_health_and_home(self):
        self.assertEqual(self.c.get("/health").get_json()["status"], "ok")
        self.assertEqual(self.c.get("/").status_code, 200)

    def test_model_missing_gives_clear_error(self):
        r = self.c.post("/diagnose", data={"photo": (jpeg_bytes(), "leaf.jpg")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 503)
        self.assertIn(b"not been trained", r.data)

    def test_rejects_non_image(self):
        r = self.c.post("/diagnose", data={"photo": (io.BytesIO(b"hello"), "leaf.jpg")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_rejects_wrong_extension(self):
        r = self.c.post("/diagnose", data={"photo": (io.BytesIO(b"x"), "leaf.exe")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_full_flow_dashboard_feedback_csv(self):
        with mock.patch.object(predictor, "predict", return_value=FAKE):
            r = self.c.post("/diagnose", data={"photo": (jpeg_bytes(), "leaf.jpg"), "location": "  mazowe ",
                                               "lat": "-17.5", "lon": "30.9"}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Maize streak virus", r.data)
        self.assertEqual(self.c.get("/dashboard").status_code, 302)  # needs sign-in
        bad = self.c.post("/login", data={"password": "nope"})
        self.assertIn(b"Wrong password", bad.data)
        self.c.post("/login", data={"password": "secret"})
        dash = self.c.get("/dashboard")
        self.assertEqual(dash.status_code, 200)
        self.assertIn(b"Mazowe", dash.data)
        self.c.post("/feedback/1", data={"value": "correct"})
        self.assertIn(b"100%", self.c.get("/dashboard").data)
        csv_resp = self.c.get("/export.csv")
        self.assertIn(b"maize_streak_virus", csv_resp.data)

    def test_api_json(self):
        with mock.patch.object(predictor, "predict", return_value=FAKE):
            r = self.c.post("/api/predict", data={"photo": (jpeg_bytes(), "a.jpg")}, content_type="multipart/form-data")
        self.assertEqual(r.get_json()["label"], "maize_streak_virus")


if __name__ == "__main__":
    unittest.main()
