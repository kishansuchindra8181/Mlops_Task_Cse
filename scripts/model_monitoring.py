import csv
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/artifacts/test_predictions.csv"
BASELINE = ROOT / "data/artifacts/evaluation_metrics.json"
OUTPUT = ROOT / "data/artifacts"

REQUIRED = [
    "image_path", "filename", "label",
    "actual_label", "predicted_label"
]
CLASSES = ["Bengin", "Malignant", "Normal"]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def calculate_metrics(rows):
    matrix = {a: {p: 0 for p in CLASSES} for a in CLASSES}

    for row in rows:
        actual = row["actual_label"]
        predicted = row["predicted_label"]
        if actual in CLASSES and predicted in CLASSES:
            matrix[actual][predicted] += 1

    total = sum(matrix[a][p] for a in CLASSES for p in CLASSES)
    accuracy = (
        sum(matrix[c][c] for c in CLASSES) / total if total else 0
    )

    scores = {}
    for c in CLASSES:
        tp = matrix[c][c]
        fp = sum(matrix[a][c] for a in CLASSES if a != c)
        fn = sum(matrix[c][p] for p in CLASSES if p != c)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )
        scores[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[c].values())
        }

    macro_f1 = sum(scores[c]["f1"] for c in CLASSES) / len(CLASSES)
    macro_precision = sum(scores[c]["precision"] for c in CLASSES) / len(CLASSES)
    macro_recall = sum(scores[c]["recall"] for c in CLASSES) / len(CLASSES)

    return {
        "samples": total,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": scores,
        "confusion_matrix": matrix
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    with INPUT.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing_columns = [c for c in REQUIRED if c not in columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_values = {
        col: sum(not (r.get(col) or "").strip() for r in rows)
        for col in REQUIRED
    }

    duplicate_ids = Counter(
        (r.get("sha256") or r.get("image_path") or "").strip()
        for r in rows
    )
    duplicate_count = sum(
        count - 1 for key, count in duplicate_ids.items()
        if key and count > 1
    )

    distribution = dict(Counter(
        (r.get("actual_label") or "").strip() for r in rows
    ))

    valid_rows = [
        r for r in rows
        if r.get("actual_label") in CLASSES
        and r.get("predicted_label") in CLASSES
    ]

    metrics = calculate_metrics(valid_rows)

    validation = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(INPUT.relative_to(ROOT)),
        "batch_type": "REPLAY_OF_EXISTING_LABELLED_TEST_PREDICTIONS",
        "total_rows": len(rows),
        "required_columns": REQUIRED,
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "duplicate_records": duplicate_count,
        "invalid_label_rows": len(rows) - len(valid_rows),
        "target_distribution": distribution,
        "validation_passed": (
            not missing_columns
            and not any(missing_values.values())
            and len(valid_rows) == len(rows)
        )
    }

    save_json(OUTPUT / "monitoring_validation.json", validation)
    save_json(OUTPUT / "incoming_batch_metrics.json", metrics)

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    champion_f1 = baseline["macro_f1_score"]
    champion_accuracy = baseline["test_accuracy"]

    # No candidate is available because the source images are missing.
    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "champion_model": "models/chest_cancer_model.keras",
        "champion_accuracy": champion_accuracy,
        "champion_macro_f1": champion_f1,
        "monitoring_batch_macro_f1": metrics["macro_f1"],
        "monitoring_batch_accuracy": metrics["accuracy"],
        "candidate_model": None,
        "candidate_training_status": "BLOCKED",
        "promotion_status": "HOLD",
        "promotion_eligible": False,
        "reason": (
            "Candidate retraining was not performed because the original "
            "image dataset is unavailable in this Codespace. The monitoring "
            "input reuses existing labelled test predictions and is not a "
            "new independent incoming dataset. Champion remains unchanged."
        ),
        "promotion_metric": "Macro F1",
        "metric_justification": (
            "Macro F1 gives equal importance to Bengin, Malignant and Normal "
            "classes, which is useful when class counts are imbalanced."
        )
    }

    save_json(OUTPUT / "quality_gate_decision.json", decision)

    print("\n===== DATA VALIDATION =====")
    print(json.dumps(validation, indent=2))
    print("\n===== MONITORING METRICS =====")
    print(json.dumps(metrics, indent=2))
    print("\n===== QUALITY GATE =====")
    print(json.dumps(decision, indent=2))
    print("\nReports saved in data/artifacts/")


if __name__ == "__main__":
    main()
