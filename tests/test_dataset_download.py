import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import download_dataset  # noqa: E402


class FakeResponse:
    def __init__(self, content, status=200):
        self.content = content
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size=-1):
        content, self.content = self.content, b""
        return content

    def getcode(self):
        return self.status


class DatasetDownloadTests(unittest.TestCase):
    def test_archive_classification(self):
        self.assertEqual(download_dataset.classify_archive("Healthy_1.rar"), "healthy")
        self.assertEqual(download_dataset.classify_archive("MLB_2.rar"), "maize_leaf_blight")
        self.assertEqual(download_dataset.classify_archive("MSV_1.rar"), "maize_streak_virus")
        self.assertIsNone(download_dataset.classify_archive("Maize_UG_Image_datasheet.pdf"))

    def test_get_dataset_files_uses_public_dataverse_metadata(self):
        payload = {
            "status": "OK",
            "data": [
                {"dataFile": {"id": 1, "filename": "Healthy_1.rar", "filesize": 12, "md5": "a" * 32}},
                {"dataFile": {"id": 2, "filename": "Maize_UG_Image_datasheet.pdf", "filesize": 5, "md5": "b" * 32}},
            ],
        }
        with mock.patch.object(
            download_dataset,
            "urlopen",
            return_value=FakeResponse(json.dumps(payload).encode("utf-8")),
        ) as open_url:
            files = download_dataset.get_dataset_files(download_dataset.DEFAULT_SERVER)

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["id"], 1)
        self.assertEqual(files[0]["class_dir"], "healthy")
        self.assertIn(
            "/api/datasets/:persistentId/versions/:latest-published/files?",
            open_url.call_args.args[0].full_url,
        )

    def test_download_saves_only_checksum_verified_archive(self):
        content = b"sample archive content"
        archive = {
            "id": 42,
            "filename": "Healthy_1.rar",
            "size": len(content),
            "md5": hashlib.md5(content).hexdigest(),
            "class_dir": "healthy",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(download_dataset, "urlopen", return_value=FakeResponse(content)):
                download_dataset.download_archive(download_dataset.DEFAULT_SERVER, archive, temp_dir)

            path = os.path.join(temp_dir, "healthy", "Healthy_1.rar")
            with open(path, "rb") as downloaded:
                self.assertEqual(downloaded.read(), content)
            self.assertFalse(os.path.exists(path + ".part"))

    def test_download_resumes_partial_archive(self):
        content = b"some downloaded archive bytes"
        partial_content = content[:10]
        archive = {
            "id": 42,
            "filename": "Healthy_1.rar",
            "size": len(content),
            "md5": hashlib.md5(content).hexdigest(),
            "class_dir": "healthy",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            class_dir = os.path.join(temp_dir, "healthy")
            os.makedirs(class_dir)
            partial_path = os.path.join(class_dir, "Healthy_1.rar.part")
            with open(partial_path, "wb") as partial:
                partial.write(partial_content)

            with mock.patch.object(
                download_dataset,
                "urlopen",
                return_value=FakeResponse(content[len(partial_content):], status=206),
            ) as open_url:
                download_dataset.download_archive(download_dataset.DEFAULT_SERVER, archive, temp_dir)

            self.assertEqual(open_url.call_args.args[0].get_header("Range"), f"bytes={len(partial_content)}-")
            with open(os.path.join(class_dir, "Healthy_1.rar"), "rb") as downloaded:
                self.assertEqual(downloaded.read(), content)


if __name__ == "__main__":
    unittest.main()
