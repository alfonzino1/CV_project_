.PHONY: help install lint test docker-build docker-up docker-down train serve deploy monitor clean run

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -e ".[dev]"
	pre-commit install 2>/dev/null || true

lint: ## Run all linters
	black src tests --check
	flake8 src tests
	isort src tests --check-only
	mypy src

format: ## Format code with black and isort
	black src tests
	isort src tests

test: ## Run pytest with coverage
	pytest tests -v --cov=src --cov-report=html

test-fast: ## Run fast tests only (no slow integration tests)
	pytest tests -v -m "not slow"

docker-build-training: ## Build training Docker image
	docker build -f docker/Dockerfile.training -t industrial-qc-training:latest .

docker-build-serving: ## Build serving Docker image
	docker build -f docker/Dockerfile.serving -t industrial-qc-serving:latest .

docker-build: docker-build-training docker-build-serving ## Build all Docker images

docker-up: ## Start local stack (MinIO, MLflow, Prefect, Prometheus, Grafana)
	docker-compose -f docker/docker-compose.yml up -d

docker-down: ## Stop local stack
	docker-compose -f docker/docker-compose.yml down

docker-logs: ## Show logs from all containers
	docker-compose -f docker/docker-compose.yml logs -f

train: ## Run training pipeline locally
	python -m src.pipelines.main_flow

train-local: ## Run training without orchestration
	python -m src.training.train

hpo: ## Run hyperparameter optimization
	python -m src.training.hpo

evaluate: ## Run evaluation pipeline
	python -m src.evaluation.metrics

serve: ## Start TorchServe locally
	torchserve --start --ts-config configs/serving.yaml --model-store models --models defect_detector.mar

serve-stop: ## Stop TorchServe
	torchserve --stop

export-model: ## Export model to TorchScript/ONNX
	python -m src.serving.archive_creator

deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/namespace.yaml
	helm upgrade --install industrial-qc ./helm -f helm/values.yaml --namespace industrial-qc

deploy-dry-run: ## Dry-run Helm deployment
	helm upgrade --install industrial-qc ./helm -f helm/values.yaml --namespace industrial-qc --dry-run --debug

rollback: ## Rollback to previous deployment
	helm rollback industrial-qc -n industrial-qc

monitor: ## Start monitoring stack (Prometheus + Grafana)
	kubectl apply -f k8s/monitoring.yaml

logs: ## Stream application logs from K8s
	kubectl logs -l app=industrial-qc -n industrial-qc -f

port-forward: ## Port-forward services for local access
	kubectl port-forward svc/industrial-qc-service 8080:8080 -n industrial-qc &
	kubectl port-forward svc/mlflow 5000:5000 -n industrial-qc &
	kubectl port-forward svc/grafana 3000:3000 -n industrial-qc &
	kubectl port-forward svc/prometheus 9090:9090 -n industrial-qc &

clean: ## Clean all generated files
	rm -rf models/ data/raw/* data/processed/* data/splits/* mlruns/ .pytest_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

run: docker-up install ## Full setup: start stack and install dependencies
	@echo "✅ Setup complete! Run 'make train' to start the pipeline."

dvc-init: ## Initialize DVC with S3 remote
	dvc init
	dvc remote add -d storage $(DVC_REMOTE)
	dvc remote modify storage region $(DVC_REMOTE_REGION)

dvc-pull: ## Pull data from DVC remote
	dvc pull

dvc-push: ## Push data to DVC remote
	dvc push

mlflow-server: ## Start MLflow server locally
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

minio-create-bucket: ## Create MinIO bucket
	mc alias set myminio $(MINIO_ENDPOINT) $(MINIO_ACCESS_KEY) $(MINIO_SECRET_KEY)
	mc mb myminio/$(MINIO_BUCKET) --ignore-existing

validate: ## Run all validation checks
	make lint
	make test
	make docker-build

ci: validate docker-build ## Simulate CI pipeline locally

cd: validate docker-build deploy ## Simulate CD pipeline locally
