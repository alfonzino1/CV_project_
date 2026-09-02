industrial-qc-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Lint, Test, Build
│       └── cd.yml                  # Deploy to K8s
├── configs/
│   ├── data.yaml                   # Dataset paths, classes
│   ├── model.yaml                  # YOLOv8 config, hyperparams
│   ├── training.yaml               # Training loop settings
│   └── serving.yaml                # TorchServe batch, timeout
├── data/
│   ├── raw/                        # Raw images (gitignored)
│   ├── processed/                  # Preprocessed datasets
│   └── splits/                     # Train/val/test splits
├── docker/
│   ├── Dockerfile.training         # Multi-stage build for training
│   ├── Dockerfile.serving          # Optimized for inference
│   └── docker-compose.yml          # Local stack (MinIO, MLflow, Prefect)
├── helm/
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── hpa.yaml
│       ├── ingress.yaml
│       └── configmap.yaml
├── k8s/
│   ├── namespace.yaml
│   ├── minio.yaml
│   ├── mlflow.yaml
│   └── monitoring.yaml             # Prometheus/Grafana
├── src/
│   ├── __init__.py
│   ├── config.py                   # Pydantic settings management
│   ├── logger.py                   # Structlog JSON formatter
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py            # S3 download, DVC pull, validation
│   │   └── preprocessing.py        # Albumentations, YOLO format conversion
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                # PyTorch Lightning + YOLOv8
│   │   └── hpo.py                  # Optuna integration
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # mAP, F1, Confusion Matrix
│   │   └── drift.py                # Evidently AI for image drift
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── handler.py              # TorchServe custom handler
│   │   └── archive_creator.py      # Create .mar file
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── collector.py            # Prometheus metrics exporter
│   │   └── alerts.py               # Drift alerting logic
│   └── pipelines/
│       ├── __init__.py
│       └── main_flow.py            # Prefect orchestration
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_data.py
│   ├── test_training.py
│   ├── test_serving.py
│   └── test_monitoring.py
├── models/                         # Local model artifacts (gitignored)
├── .dvc/
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── PROJECT_STRUCTURE.md
