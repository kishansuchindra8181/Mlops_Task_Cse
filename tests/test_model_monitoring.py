import csv
import tempfile
import unittest
from pathlib import Path

from scripts.model_monitoring import (
    calculate_metrics,
    quality_gate,
    validate_batch,
)


class ModelMonitoringTests(unittest.TestCase):
    def test_metrics_include_all_classes_and_macro_f1(self):
        result = calculate_metrics(
            ["Bengin", "Bengin", "Bengin", "Malignant", "Normal"],
            ["Bengin", "Bengin", "Bengin", "Malignant", "Malignant"],
        )

        self.assertEqual(result["samples"], 5)
        self.assertEqual(result["accuracy"], 0.8)
        self.assertEqual(result["confusion_matrix"]["Normal"]["Malignant"], 1)
        self.assertLess(result["macro_f1"], result["accuracy"])

    def test_batch_validation_detects_duplicate_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_a = root / "a.jpg"
            image_b = root / "b.jpg"
            image_a.write_bytes(b"same image")
            image_b.write_bytes(b"same image")
            batch = root / "batch.csv"
            with batch.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["image_path", "label"])
                writer.writeheader()
                writer.writerows([
                    {"image_path": "a.jpg", "label": "Bengin"},
                    {"image_path": "b.jpg", "label": "Normal"},
                ])

            _, report = validate_batch(batch, root=root)

        self.assertEqual(report["total_rows"], 2)
        self.assertEqual(report["duplicate_records"], 1)
        self.assertFalse(report["validation_passed"])

    def test_batch_validation_reports_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            batch = Path(temp_dir) / "batch.csv"
            batch.write_text("filename\nimage.jpg\n", encoding="utf-8")

            rows, report = validate_batch(batch, root=Path(temp_dir))

        self.assertEqual(rows, [])
        self.assertEqual(report["missing_columns"], ["image_path", "label"])
        self.assertFalse(report["validation_passed"])

    def test_quality_gate_only_marks_candidate_eligible_when_macro_f1_not_lower(self):
        champion = {"macro_f1": 0.8}
        self.assertTrue(
            quality_gate(champion, {"macro_f1": 0.81})["promotion_eligible"]
        )
        self.assertFalse(
            quality_gate(champion, {"macro_f1": 0.79})["promotion_eligible"]
        )


if __name__ == "__main__":
    unittest.main()
