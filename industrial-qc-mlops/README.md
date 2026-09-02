# Industrial QC MLOps Platform

Production-ready MLOps pipeline for industrial defect detection using YOLOv8.

## Quick Start

```bash
make run
```

## Features

- **Data Versioning**: DVC + MinIO/S3
- **Experiment Tracking**: MLflow
- **Orchestration**: Prefect 2.0
- **Training**: YOLOv8 with Optuna HPO
- **Serving**: TorchServe with dynamic batching (<50ms latency)
- **Monitoring**: Prometheus + Grafana + Evidently AI
- **Deployment**: Kubernetes (Helm) + GitHub Actions CI/CD

## Project Structure

See `PROJECT_STRUCTURE.md` for detailed layout.

## Requirements

- Python 3.10+
- Docker + NVIDIA Container Toolkit
- Kubernetes cluster (minikube/kind for local, EKS/GKE for prod)
- GPU: RTX 3090 or equivalent (for training)

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key environment variables:
- `MINIO_ENDPOINT`: S3-compatible storage
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `DVC_REMOTE`: DVC remote storage URL
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: Credentials

## Makefile Commands

- `make install` - Install dependencies
- `make lint` - Run linters (black, flake8, mypy)
- `make test` - Run pytest
- `make docker-build` - Build Docker images
- `make docker-up` - Start local stack (MinIO, MLflow, Prefect)
- `make train` - Run training pipeline
- `make serve` - Start TorchServe locally
- `make deploy` - Deploy to Kubernetes
- `make monitor` - Start Prometheus + Grafana

## Pipeline Stages

1. **Data Ingestion**: Download from S3, validate schema, DVC pull
2. **Preprocessing**: Albumentations, YOLO format conversion
3. **Training**: YOLOv8 fine-tuning with Optuna HPO
4. **Evaluation**: mAP, F1, confusion matrix, drift detection
5. **Export**: TorchScript/ONNX with INT8 quantization
6. **Serving**: TorchServe REST/gRPC API
7. **Monitoring**: Latency, throughput, data drift alerts

## Monitoring Dashboards

Access Grafana at `http://localhost:3000` (admin/admin).

Pre-configured dashboards:
- Model performance (mAP, precision, recall)
- Inference latency & throughput
- Data drift (PSI, KL-divergence)
- System metrics (GPU memory, CPU usage)

## License

MIT
