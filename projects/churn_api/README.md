# Customer Churn Prediction API

## 🧠 System Overview
This project demonstrates an end-to-end AI Systems workflow:
- **Model Versioning:** Each training run saves model + metadata with Git SHA & metrics (e.g., `models/churn_model_v1.1.pkl`, `models/metadata_v1.1.json`).
- **API Service:** Version-aware FastAPI app serving predictions at `/predict`, plus `/health` and `/version` for ops visibility.
- **Logging:** Every prediction is appended to `logs/prediction_log.jsonl` (timestamp, model version, request, response) for monitoring & future retraining.
- **Containerization:** Fully Dockerized with Docker Compose support for one-command local runs.

## 🚀 Run Locally

### Prerequisites
- Docker & Docker Compose installed
- A trained model exists in `models/` (e.g., `churn_model_v1.1.pkl`). If not, train first:
  ```bash
  # inside a virtualenv
  MODEL_VERSION=v1.1 python src/train.py --data data/churn_sample.csv
