import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def test_class_folder_names_are_canonicalized_without_merging_distinct_blight_classes(self):
        expected = {
            "Common rust": "common_rust",
            "Gray leaf spot": "gray_leaf_spot",
            "Healthy": "healthy",
            "Magnessium": "magnesium_deficiency",
            "Maize streak virus": "maize_streak_virus",
            "Nitrogen": "nitrogen_deficiency",
            "Northern Leaf Blight": "northern_leaf_blight",
            "Phosphorous": "phosphorus_deficiency",
            "Pottasium": "potassium_deficiency",
            "Maize Leaf Blight": "maize_leaf_blight",
        }
        for folder, label in expected.items():
            with self.subTest(folder=folder):
                self.assertEqual(prepare_dataset.canon(folder), label)

    def test_folder_dataset_splits_all_recognized_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "dataset")
            destination = os.path.join(temp_dir, "dataset_split")
            folder_labels = {
                "Common rust": "common_rust",
                "Northern Leaf Blight": "northern_leaf_blight",
                "Magnessium": "magnesium_deficiency",
            }
            for folder_index, folder in enumerate(folder_labels):
                class_dir = os.path.join(source, folder)
                os.makedirs(class_dir)
                for image_index in range(20):
                    color = (folder_index * 70, image_index * 10, 100)
                    Image.new("RGB", (8, 8), color).save(
                        os.path.join(class_dir, f"{image_index}.jpg")
                    )

            with mock.patch(
                "sys.argv",
                ["prepare_dataset.py", "--src", source, "--dst", destination],
            ):
                prepare_dataset.main()

            for split in ("train", "val", "test"):
                for label in folder_labels.values():
                    split_dir = os.path.join(destination, split, label)
                    self.assertTrue(os.path.isdir(split_dir))
                    self.assertGreater(len(os.listdir(split_dir)), 0)


if __name__ == "__main__":
    unittest.main()
