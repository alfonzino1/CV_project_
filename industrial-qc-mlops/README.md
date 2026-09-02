# Industrial Quality Control MLOps Platform

## 🏭 Industrial Defect Detection System

Production-ready MLOps pipeline for automated visual quality control in manufacturing.
Detects defects on metal surfaces: cracks, inclusions, stains, pitting, and scale.

### Key Features
- **Real-time Inference**: <50ms latency with TorchServe + Dynamic Batching
- **Data Versioning**: DVC + MinIO/S3 for reproducible datasets
- **Experiment Tracking**: MLflow with auto-logging
- **Hyperparameter Optimization**: Optuna integration
- **Drift Detection**: Evidently AI for visual data drift (lighting, angles, weather)
- **Scalable Deployment**: Kubernetes with Helm charts
- **CI/CD**: GitHub Actions for automated testing and deployment

### Tech Stack
- **Model**: YOLOv8 (PyTorch)
- **Framework**: PyTorch Lightning
- **Serving**: TorchServe
- **Orchestration**: Prefect 2.0
- **Monitoring**: Prometheus + Grafana + Evidently AI
- **Infra**: Docker, Kubernetes, MinIO

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA 11.8+
- Make installed

### One-Liner Setup
```bash
make run
```

This command starts the full local stack:
- MinIO (S3-compatible storage)
- MLflow (Tracking & Registry)
- Prefect (Orchestration)
- Prometheus + Grafana (Monitoring)

### Step-by-Step Guide

#### 1. Clone and Configure
```bash
git clone <repo-url>
cd industrial-qc-mlops
cp .env.example .env
# Edit .env with your credentials
```

#### 2. Initialize Data
```bash
make data-init
make data-pull
```

#### 3. Train Model
```bash
make train
# Or with hyperparameter optimization
make hpo
```

#### 4. Serve Model
```bash
make serve
# Test inference
curl -X POST http://localhost:8080/predictions/defect_detector -T test_image.jpg
```

#### 5. Monitor
Open Grafana at `http://localhost:3000` (admin/admin).

---

## 📂 Project Structure

```
industrial-qc-mlops/
├── configs/                # Configuration files
│   ├── data.yaml           # Dataset config
│   ├── model.yaml          # Model hyperparameters
│   ├── training.yaml       # Training settings
│   └── serving.yaml        # Serving config
├── data/                   # Data directories (DVC tracked)
│   ├── raw/                # Raw images
│   ├── processed/          # Preprocessed data
│   └── validated/          # Validated splits
├── src/                    # Source code
│   ├── config.py           # Config management (Pydantic)
│   ├── logger.py           # Structured logging
│   ├── data/
│   │   ├── ingestion.py    # Data loading & validation
│   │   └── preprocessing.py# Augmentations & transforms
│   ├── training/
│   │   ├── train.py        # Training loop
│   │   └── hpo.py          # Hyperparameter optimization
│   ├── evaluation/
│   │   ├── metrics.py      # mAP, F1, Confusion Matrix
│   │   └── drift.py        # Data drift detection
│   └── serving/
│       ├── handler.py      # TorchServe handler
│       └── archive_creator.py # Model export
├── docker/                 # Dockerfiles & compose
│   ├── Dockerfile.training
│   ├── Dockerfile.serving
│   └── docker-compose.yml
├── helm/                   # Kubernetes Helm charts
├── tests/                  # Pytest tests
├── .github/workflows/      # CI/CD pipelines
├── Makefile                # Common commands
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠️ Makefile Commands

| Command | Description |
|---------|-------------|
| `make run` | Start full local stack |
| `make stop` | Stop all services |
| `make data-init` | Initialize DVC and MinIO buckets |
| `make data-pull` | Pull dataset from remote storage |
| `make train` | Run training pipeline |
| `make hpo` | Run hyperparameter optimization |
| `make evaluate` | Evaluate model and log metrics |
| `make serve` | Start TorchServe locally |
| `make benchmark` | Run inference benchmarks |
| `make deploy` | Deploy to Kubernetes |
| `make monitor` | Open Grafana dashboards |
| `make test` | Run all tests |
| `make lint` | Run linters (black, flake8, mypy) |

---

## 📊 Monitoring & Alerts

### Metrics Tracked
- **System**: Latency, Throughput, GPU Memory, CPU Usage
- **Model**: mAP@0.5, Precision, Recall, F1-Score
- **Data Drift**: PSI (Population Stability Index), KL-Divergence
- **Business**: Defect Rate, False Positive/Negative Rates

### Drift Detection
Automated alerts when:
- Lighting conditions change significantly
- Camera angles shift
- New defect types appear
- Image quality degrades

---

## 🔐 Security

- All secrets via environment variables (`.env`)
- No hardcoded credentials
- RBAC configured for Kubernetes
- Network policies for service isolation

---

## 🤝 Contributing

1. Create a feature branch
2. Run `make lint` and `make test`
3. Submit a PR

---

## 📄 License

MIT License

---

## 📞 Support

For issues or questions, open a GitHub issue or contact the MLOps team.
