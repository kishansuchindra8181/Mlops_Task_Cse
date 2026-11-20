# Chest Cancer Classification – End-to-End MLOps

An end-to-end machine learning project that classifies lung CT scan images into three categories: **Bengin, Malignant, and Normal**. The project combines deep learning, FastAPI, Docker, GitHub Actions, and Docker Hub to demonstrate an automated ML deployment workflow.

## Project Overview

This project uses a transfer-learning approach with MobileNetV2 to classify lung CT scan images. A FastAPI application exposes the trained model through REST API endpoints.

The application is containerized using Docker, while GitHub Actions automates testing, Docker image building, and publishing to Docker Hub.

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- FastAPI
- Docker
- GitHub Actions
- Docker Hub
- Pytest
- Scikit-learn

## Dataset

**IQ-OTHNCCD Lung Cancer Dataset**

Three classification categories:
- Bengin
- Malignant
- Normal

Input image size: `160 × 160`

## Model Architecture

- MobileNetV2 pretrained backbone
- Frozen base model
- Global Average Pooling
- Dropout (0.3)
- Dense layer (128 neurons, ReLU)
- Output layer (3 neurons, Softmax)

Optimizer: Adam  
Learning rate: 0.001  
Loss: Sparse Categorical Crossentropy  
Training epochs: 8

## Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 94.97% |
| Macro F1 Score | 89.54% |
| Classification Errors | 8 |

These results are from the project's test evaluation and are not evidence of clinical performance.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Predict from an uploaded image |

Interactive API documentation:

`http://localhost:8000/docs`

## Run with Docker

Pull the published image:

```bash
docker pull deepakchaudhary11/chest-cancer-mlops:latest
```

Run the container:

```bash
docker run -d -p 8000:8080 --name chest-cancer-app deepakchaudhary11/chest-cancer-mlops:latest
```

Open:

`http://localhost:8000/docs`

Docker Hub image:

`deepakchaudhary11/chest-cancer-mlops:latest`

## CI/CD Pipeline

GitHub Actions automates the following workflow:

1. Checkout source code.
2. Set up Python 3.11.
3. Install project dependencies.
4. Run API tests using Pytest.
5. Build the Docker image.
6. Authenticate with Docker Hub using GitHub repository secrets.
7. Push the image with `latest` and commit SHA tags.

Required GitHub repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## Run Tests Locally

```bash
pytest -v
```

## Repository Structure

```text
chest-cancer-mlops/
│
├── api/
│   └── main.py
├── config/
├── data/
│   └── artifacts/
├── models/
│   ├── chest_cancer_model.keras
│   └── chest_cancer_savedmodel/
├── src/
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pytest.ini
├── requirements.txt
├── setup.py
└── README.md
```

## MLOps Practical Assessment: Monitoring and Retraining

The workflow in `scripts/model_monitoring.py` validates a labelled incoming
image batch, evaluates the current champion on that batch, trains a separate
candidate from the existing training split plus the new batch, and compares
the champion and candidate on the unchanged held-out test split. It writes
validation, prediction, evaluation, and quality-gate reports under
`data/artifacts/`. If MLflow is installed, it also records the candidate run;
otherwise the decision report records that tracking was unavailable.

Prepare `data/incoming/batch.csv` with these columns:

```csv
image_path,label
data/raw/new-images/image-001.jpg,Malignant
data/raw/new-images/image-002.jpg,Normal
```

Paths may be absolute or relative to the repository root. Provide the original
images as well as the existing `data/processed/train.csv`, `val.csv`, and
`test.csv` splits and `models/chest_cancer_model.keras`. Then run:

```bash
python scripts/model_monitoring.py --batch data/incoming/batch.csv
```

The batch is checked for missing values, invalid labels, missing image files,
and duplicate image contents. Images already found in an existing train,
validation, or test split are excluded from retraining to prevent leakage.

Macro F1 is the promotion metric because it weights all three classes equally
despite their different sample counts. The candidate is only marked eligible
when its macro F1 on the held-out test split is at least the champion's. The
workflow never replaces the champion automatically; any promotion is a
separate, recorded action.

The checked-in monitoring artifacts from the earlier replay are explicitly
marked as using existing test predictions, not a new independent batch. They
are not evidence of a completed candidate-training run. To generate assessment
execution evidence, supply the incoming images and training splits above and
run the workflow in an environment with the project's TensorFlow dependencies.

## Future Improvements

- Data drift detection
- Cloud deployment
- Model explainability using Grad-CAM
- Improved validation on additional independent datasets

## Disclaimer

This project is developed for educational and research purposes. It is not a clinically validated diagnostic system and must not be used to make medical decisions. Always consult qualified healthcare professionals.

## Author

**Deepak Chaudhary**

GitHub: https://github.com/CodewithDeepakChaudhary

Project: https://github.com/CodewithDeepakChaudhary/chest-cancer-mlops
