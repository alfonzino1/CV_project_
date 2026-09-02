"""
Prefect 2.0 orchestrations for the Industrial QC MLOps pipeline.
Defines reusable flows and tasks for data ingestion, training, evaluation, and deployment.
"""
from prefect import flow, task, get_run_logger
from prefect.logging import get_logger
from pathlib import Path
from typing import Optional, Dict, Any

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.training.train import train_model
from src.evaluation.metrics import evaluate_model
from src.serving.archive_creator import export_model
from src.config import get_settings


@task(log_prints=True)
def ingest_data_task(data_source: str, output_dir: str) -> Dict[str, Any]:
    """Task: Download and validate dataset."""
    logger = get_run_logger()
    logger.info(f"Ingesting data from {data_source}")
    
    ingestion = DataIngestion()
    result = ingestion.download_from_s3(data_source, output_dir)
    
    validation = ingestion.validate_dataset_structure(Path(output_dir))
    
    return {"download": result, "validation": validation}


@task(log_prints=True)
def preprocess_data_task(
    raw_dir: str,
    processed_dir: str,
    image_size: int = 640,
    augment: bool = True
) -> Dict[str, Any]:
    """Task: Apply augmentations and create dataset splits."""
    logger = get_run_logger()
    logger.info(f"Preprocessing data from {raw_dir}")
    
    preprocessor = DataPreprocessor(image_size=image_size)
    result = preprocessor.process_directory(
        raw_dir=Path(raw_dir),
        output_dir=Path(processed_dir),
        augment=augment
    )
    
    return result


@task(log_prints=True)
def train_model_task(
    data_yaml: str,
    model_config: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640
) -> Dict[str, Any]:
    """Task: Train YOLOv8 model."""
    logger = get_run_logger()
    logger.info(f"Training model with config {model_config}")
    
    settings = get_settings()
    
    # Run training
    results = train_model(
        data_yaml=Path(data_yaml),
        model_name=settings.MODEL_NAME,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        project=output_dir
    )
    
    return {"model_path": results.get("best_model", ""), "metrics": results.get("metrics", {})}


@task(log_prints=True)
def evaluate_model_task(
    model_path: str,
    data_yaml: str,
    output_dir: str
) -> Dict[str, Any]:
    """Task: Evaluate trained model."""
    logger = get_run_logger()
    logger.info(f"Evaluating model {model_path}")
    
    metrics = evaluate_model(
        model_path=model_path,
        data_yaml=Path(data_yaml),
        output_dir=Path(output_dir)
    )
    
    return metrics


@task(log_prints=True)
def export_model_task(
    model_path: str,
    output_dir: str,
    formats: list = ["torchscript", "onnx"]
) -> Dict[str, str]:
    """Task: Export model to various formats."""
    logger = get_run_logger()
    logger.info(f"Exporting model to {formats}")
    
    exported_paths = export_model(
        model_path=model_path,
        output_dir=Path(output_dir),
        formats=formats
    )
    
    return exported_paths


@flow(name="industrial-qc-training-pipeline", version="1.0.0")
def training_pipeline(
    data_source: str = "s3://industrial-qc-data/raw/",
    epochs: int = 100,
    batch_size: int = 16,
    run_hpo: bool = False
):
    """
    Main training pipeline flow.
    
    Orchestrates:
    1. Data ingestion from S3/MinIO
    2. Data preprocessing & augmentation
    3. Model training (with optional HPO)
    4. Model evaluation
    5. Model export
    """
    logger = get_run_logger()
    settings = get_settings()
    
    # Paths
    raw_dir = settings.data_raw_dir
    processed_dir = settings.data_processed_dir
    models_dir = settings.models_dir
    
    # Step 1: Ingest data
    ingestion_result = ingest_data_task(data_source, raw_dir)
    if not ingestion_result["validation"]["valid"]:
        raise ValueError(f"Data validation failed: {ingestion_result['validation']['message']}")
    
    # Step 2: Preprocess data
    preprocess_result = preprocess_data_task(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        image_size=settings.IMAGE_SIZE,
        augment=True
    )
    
    # Step 3: Train model
    train_result = train_model_task(
        data_yaml=str(settings.DATA_YAML_PATH),
        model_config=str(settings.MODEL_CONFIG_PATH),
        output_dir=str(models_dir),
        epochs=epochs,
        batch_size=batch_size,
        imgsz=settings.IMAGE_SIZE
    )
    
    model_path = train_result["model_path"]
    
    # Step 4: Evaluate model
    eval_metrics = evaluate_model_task(
        model_path=model_path,
        data_yaml=str(settings.DATA_YAML_PATH),
        output_dir=str(models_dir / "evaluations")
    )
    
    logger.info(f"Evaluation metrics: {eval_metrics}")
    
    # Check if model meets threshold
    mAP50 = eval_metrics.get("mAP50", 0.0)
    if mAP50 < settings.MODEL_THRESHOLD_MAP50:
        logger.warning(f"Model mAP50 ({mAP50:.4f}) below threshold ({settings.MODEL_THRESHOLD_MAP50})")
    
    # Step 5: Export model
    exported = export_model_task(
        model_path=model_path,
        output_dir=str(models_dir / "exported"),
        formats=["torchscript", "onnx", "mar"]
    )
    
    logger.info(f"Model exported to: {exported}")
    
    return {
        "status": "success",
        "model_path": model_path,
        "metrics": eval_metrics,
        "exported_formats": exported
    }


@flow(name="industrial-qc-hpo-pipeline", version="1.0.0")
def hpo_pipeline(
    data_source: str = "s3://industrial-qc-data/raw/",
    n_trials: int = 50,
    timeout_hours: int = 12
):
    """
    Hyperparameter optimization pipeline.
    
    Uses Optuna to find optimal hyperparameters:
    - Learning rate
    - Weight decay
    - Augmentation parameters
    - Model architecture variants
    """
    logger = get_run_logger()
    logger.info(f"Starting HPO with {n_trials} trials")
    
    from src.training.hpo import run_hyperparameter_optimization
    
    # Reuse ingestion and preprocessing tasks
    settings = get_settings()
    raw_dir = settings.data_raw_dir
    processed_dir = settings.data_processed_dir
    
    ingestion_result = ingest_data_task(data_source, raw_dir)
    preprocess_result = preprocess_data_task(raw_dir, processed_dir)
    
    # Run HPO
    best_params = run_hyperparameter_optimization(
        data_yaml=settings.DATA_YAML_PATH,
        n_trials=n_trials,
        timeout=timeout_hours * 3600
    )
    
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Retrain with best params
    # (Implementation would pass best_params to training)
    
    return {"best_params": best_params}


if __name__ == "__main__":
    # Run training pipeline
    training_pipeline.serve(name="industrial-qc-training")
