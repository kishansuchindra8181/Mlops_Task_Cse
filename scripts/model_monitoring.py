import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Bengin", "Malignant", "Normal"]
REQUIRED_COLUMNS = ["image_path", "label"]
ARTIFACT_DIR = ROOT / "data" / "artifacts"
CHAMPION_PATH = ROOT / "models" / "chest_cancer_model.keras"
CANDIDATE_PATH = ROOT / "models" / "candidate_chest_cancer_model.keras"


def save_json(path, data):
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def calculate_metrics(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must have equal lengths")
    if not actual:
        raise ValueError("Cannot calculate metrics for an empty batch")
    if any(label not in CLASSES for label in actual + predicted):
        raise ValueError("Metrics contain an unknown class label")

    matrix = {label: {guess: 0 for guess in CLASSES} for label in CLASSES}
    for label, guess in zip(actual, predicted):
        matrix[label][guess] += 1

    per_class = {}
    for label in CLASSES:
        true_positive = matrix[label][label]
        false_positive = sum(
            matrix[other][label] for other in CLASSES if other != label
        )
        false_negative = sum(
            matrix[label][other] for other in CLASSES if other != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }

    return {
        "samples": len(actual),
        "accuracy": sum(matrix[label][label] for label in CLASSES) / len(actual),
        "macro_precision": sum(
            per_class[label]["precision"] for label in CLASSES
        ) / len(CLASSES),
        "macro_recall": sum(
            per_class[label]["recall"] for label in CLASSES
        ) / len(CLASSES),
        "macro_f1": sum(per_class[label]["f1"] for label in CLASSES) / len(CLASSES),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _resolve_image_path(value, root):
    normalized = value.strip().replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_batch(csv_path, root=ROOT):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Incoming batch CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    missing_values = {
        column: sum(not (row.get(column) or "").strip() for row in rows)
        for column in REQUIRED_COLUMNS
    }
    if missing_columns:
        return [], {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": str(csv_path),
            "total_rows": len(rows),
            "required_columns": REQUIRED_COLUMNS,
            "missing_columns": missing_columns,
            "missing_values": missing_values,
            "invalid_label_rows": [],
            "missing_images": [],
            "duplicate_records": 0,
            "target_distribution": {},
            "validation_passed": False,
        }

    invalid_labels = [
        index for index, row in enumerate(rows, start=2)
        if (row.get("label") or "").strip() not in CLASSES
    ]
    target_distribution = dict(Counter(
        (row.get("label") or "").strip() for row in rows
        if (row.get("label") or "").strip()
    ))

    missing_images = []
    duplicate_hashes = Counter()
    validated_rows = []
    for row_number, row in enumerate(rows, start=2):
        image_value = (row.get("image_path") or "").strip()
        if not image_value:
            continue
        image_path = _resolve_image_path(image_value, root)
        if not image_path.is_file():
            missing_images.append({"row": row_number, "image_path": image_value})
            continue
        image_hash = _sha256(image_path)
        duplicate_hashes[image_hash] += 1
        validated_rows.append({
            **row,
            "image_path": str(image_path),
            "label": (row.get("label") or "").strip(),
            "sha256": image_hash,
        })

    duplicate_records = sum(
        count - 1 for count in duplicate_hashes.values() if count > 1
    )
    validation = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(csv_path),
        "total_rows": len(rows),
        "required_columns": REQUIRED_COLUMNS,
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "invalid_label_rows": invalid_labels,
        "missing_images": missing_images,
        "duplicate_records": duplicate_records,
        "target_distribution": target_distribution,
        "validation_passed": (
            bool(rows)
            and not any(missing_values.values())
            and not invalid_labels
            and not missing_images
            and duplicate_records == 0
        ),
    }
    return validated_rows, validation


def quality_gate(champion_metrics, candidate_metrics):
    champion_f1 = champion_metrics["macro_f1"]
    candidate_f1 = candidate_metrics["macro_f1"]
    eligible = candidate_f1 >= champion_f1
    return {
        "promotion_metric": "Macro F1",
        "metric_justification": (
            "Macro F1 weights Bengin, Malignant and Normal equally, so a "
            "large class cannot hide poor performance on a smaller class."
        ),
        "champion_macro_f1": champion_f1,
        "candidate_macro_f1": candidate_f1,
        "promotion_eligible": eligible,
        "promotion_status": (
            "ELIGIBLE_FOR_MANUAL_PROMOTION" if eligible else "RETAIN_CHAMPION"
        ),
        "reason": (
            "Candidate macro F1 meets or exceeds the champion on the held-out "
            "test set. Promotion still requires a separate manual action."
            if eligible else
            "Candidate macro F1 is below the champion on the held-out test set."
        ),
    }


def _predict_metrics(model, dataframe, create_dataset, index_to_class):
    probabilities = model.predict(create_dataset(dataframe), verbose=0)
    predicted = [index_to_class[int(index)] for index in probabilities.argmax(axis=1)]
    actual = dataframe["label"].astype(str).tolist()
    return calculate_metrics(actual, predicted), predicted


def _track_run(metrics, candidate_path, report_paths):
    try:
        import mlflow
    except ModuleNotFoundError:
        return {"status": "unavailable", "reason": "MLflow is not installed"}

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("chest-cancer-classification")
    with mlflow.start_run(run_name="monitoring-retraining-candidate") as run:
        mlflow.log_param("candidate_model", str(candidate_path.relative_to(ROOT)))
        mlflow.log_param("promotion_metric", "macro_f1")
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)
        mlflow.log_artifact(str(candidate_path))
        for report_path in report_paths:
            mlflow.log_artifact(str(report_path))
        return {"status": "logged", "run_id": run.info.run_id}


def run_workflow(batch_path, epochs=8):
    incoming_rows, validation = validate_batch(batch_path)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    validation_path = ARTIFACT_DIR / "monitoring_validation.json"
    save_json(validation_path, validation)
    if not validation["validation_passed"]:
        raise ValueError(
            f"Incoming batch validation failed; see {validation_path}"
        )
    if not CHAMPION_PATH.is_file():
        raise FileNotFoundError(f"Champion model not found: {CHAMPION_PATH}")

    # Heavy ML dependencies are loaded only after the input batch is validated.
    import pandas as pd
    import tensorflow as tf

    from src.chest_cancer.components.model_trainer import (
        CLASS_TO_INDEX,
        create_dataset,
    )

    split_dir = ROOT / "data" / "processed"
    train_path = split_dir / "train.csv"
    validation_split_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"
    for path in (train_path, validation_split_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required existing data split is missing: {path}"
            )

    incoming_df = pd.DataFrame(incoming_rows)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(validation_split_path)
    test_df = pd.read_csv(test_path)
    for name, frame in (
        ("training", train_df), ("validation", val_df), ("test", test_df)
    ):
        missing = {"image_path", "label"} - set(frame.columns)
        if (
            missing
            or frame[["image_path", "label"]].isnull().any().any()
            or frame[["image_path", "label"]].astype(str).apply(
                lambda column: column.str.strip().eq("").any()
            ).any()
        ):
            raise ValueError(f"Invalid {name} CSV; missing columns: {sorted(missing)}")
        invalid = set(frame["label"].astype(str)) - set(CLASSES)
        if invalid:
            raise ValueError(f"Unexpected labels in {name} CSV: {sorted(invalid)}")
        frame["image_path"] = frame["image_path"].astype(str).map(
            lambda value: str(_resolve_image_path(value, ROOT))
        )
        missing_paths = [
            path for path in frame["image_path"] if not Path(path).is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"{len(missing_paths)} {name} image files are missing"
            )

    # Keep incoming images out of every existing split to prevent test leakage.
    existing_hashes = set()
    existing_paths = set()
    for frame in (train_df, val_df, test_df):
        if "sha256" in frame.columns:
            existing_hashes.update(frame["sha256"].dropna().astype(str))
        existing_paths.update(frame["image_path"].astype(str))
    duplicate_mask = (
        incoming_df["sha256"].isin(existing_hashes)
        | incoming_df["image_path"].isin(existing_paths)
    )
    excluded_existing_records = int(duplicate_mask.sum())
    incoming_df = incoming_df[~duplicate_mask]
    if incoming_df.empty:
        raise ValueError("No new incoming records remain after duplicate checking")
    combined_train_df = pd.concat([train_df, incoming_df], ignore_index=True)

    index_to_class = {index: name for name, index in CLASS_TO_INDEX.items()}
    champion = tf.keras.models.load_model(CHAMPION_PATH, compile=False)
    incoming_metrics, incoming_predictions = _predict_metrics(
        champion, incoming_df, create_dataset, index_to_class
    )
    incoming_metrics_path = ARTIFACT_DIR / "incoming_batch_metrics.json"
    save_json(incoming_metrics_path, incoming_metrics)
    predictions_path = ARTIFACT_DIR / "incoming_batch_predictions.csv"
    prediction_rows = [
        {
            "image_path": row["image_path"],
            "actual_label": row["label"],
            "predicted_label": prediction,
        }
        for row, prediction in zip(
            incoming_df.to_dict("records"), incoming_predictions
        )
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)

    champion_test_metrics, _ = _predict_metrics(
        champion, test_df, create_dataset, index_to_class
    )
    candidate = tf.keras.models.load_model(CHAMPION_PATH, compile=False)
    candidate.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    candidate.fit(
        create_dataset(combined_train_df, shuffle=True),
        validation_data=create_dataset(val_df),
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2
            ),
        ],
    )
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate.save(CANDIDATE_PATH)
    candidate_test_metrics, _ = _predict_metrics(
        candidate, test_df, create_dataset, index_to_class
    )

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "champion_model": str(CHAMPION_PATH.relative_to(ROOT)),
        "candidate_model": str(CANDIDATE_PATH.relative_to(ROOT)),
        "candidate_training_status": "COMPLETED",
        "training_samples": len(combined_train_df),
        "incoming_batch_samples": len(incoming_df),
        "incoming_records_already_in_existing_splits": excluded_existing_records,
        "incoming_batch_metrics": incoming_metrics,
        "champion_test_metrics": champion_test_metrics,
        "candidate_test_metrics": candidate_test_metrics,
        **quality_gate(champion_test_metrics, candidate_test_metrics),
    }
    decision["mlflow"] = _track_run(
        {
            "champion_test_macro_f1": champion_test_metrics["macro_f1"],
            "candidate_test_macro_f1": candidate_test_metrics["macro_f1"],
            "incoming_batch_macro_f1": incoming_metrics["macro_f1"],
            "candidate_test_accuracy": candidate_test_metrics["accuracy"],
        },
        CANDIDATE_PATH,
        [validation_path, incoming_metrics_path, predictions_path],
    )
    decision_path = ARTIFACT_DIR / "quality_gate_decision.json"
    save_json(decision_path, decision)

    print(f"Validated incoming rows: {validation['total_rows']}")
    print(f"Candidate model: {CANDIDATE_PATH}")
    print(f"Quality-gate decision: {decision['promotion_status']}")
    print(f"Decision report: {decision_path}")
    print("Champion model was not overwritten.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an incoming image batch and train a gated candidate."
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=ROOT / "data" / "incoming" / "batch.csv",
        help="CSV containing image_path,label columns",
    )
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    run_workflow(args.batch, epochs=args.epochs)


if __name__ == "__main__":
    main()
