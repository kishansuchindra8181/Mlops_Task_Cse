
from pathlib import Path
import os

import mlflow
import mlflow.tensorflow
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

MODEL_PATH = ROOT / "models" / "chest_cancer_model.keras"
CSV_PATH = ROOT / "data" / "artifacts" / "test_predictions.csv"

EXPERIMENT = "chest-cancer-classification"
MODEL_NAME = "chest-cancer-classifier"

CLASSES = ["Bengin", "Malignant", "Normal"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

# Local MLflow tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT)

client = MlflowClient()

# Load existing trained model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Load actual evaluation predictions
df = pd.read_csv(CSV_PATH)

required = {"actual_label", "predicted_label"}

if not required.issubset(df.columns):
    raise ValueError(f"Missing CSV columns: {required - set(df.columns)}")

y_true = df["actual_label"].astype(str)
y_pred = df["predicted_label"].astype(str)

if not (set(y_true) | set(y_pred)).issubset(set(CLASSES)):
    raise ValueError("Unexpected class labels in prediction CSV")

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true, y_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true, y_pred,
    labels=CLASSES,
    average="weighted",
    zero_division=0
)

precision = precision_score(
    y_true, y_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true, y_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    labels=CLASSES,
    output_dict=True,
    zero_division=0
)

with mlflow.start_run(run_name="mobilenetv2-baseline") as run:

    mlflow.log_params({
        "architecture": "MobileNetV2",
        "dataset": "IQ-OTHNCCD",
        "image_size": "160x160",
        "epochs": 8,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "dropout": 0.3,
        "dense_units": 128,
        "classes": ",".join(CLASSES),
    })

    mlflow.log_metrics({
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    })

    for name in CLASSES:
        mlflow.log_metric(
            f"f1_{name.lower()}",
            float(report[name]["f1-score"])
        )

    mlflow.log_artifact(str(CSV_PATH))

    # Register the existing trained Keras model
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )

    print("MLflow run:", run.info.run_id)


# Verify registration
versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

if not versions:
    raise RuntimeError("Model registration failed")

# Find latest registered version
latest = max(versions, key=lambda v: int(v.version))

client.set_model_version_tag(
    MODEL_NAME,
    latest.version,
    "validation_status",
    "evaluated"
)

# Promote the validated baseline
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest.version,
    stage="Production",
    archive_existing_versions=True
)

print("\nMLFLOW SUCCESS")
print("Experiment:", EXPERIMENT)
print("Model:", MODEL_NAME)
print("Version:", latest.version)
print("Stage: Production")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Macro F1:", round(macro_f1 * 100, 2), "%")
