"""
Configuration management using Pydantic.

Loads configuration from YAML files and environment variables,
providing type-safe access to all settings.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseSettings, Field, validator


def _resolve_env_vars(value: str) -> str:
    """Resolve environment variables in format ${VAR_NAME}."""
    if not isinstance(value, str):
        return value
    
    pattern = r'\$\{([^}]+)\}'
    
    def replacer(match: re.Match) -> str:
        env_var = match.group(1)
        return os.environ.get(env_var, match.group(0))
    
    return re.sub(pattern, replacer, value)


def _load_yaml_with_env_resolution(config_path: Path) -> Dict[str, Any]:
    """Load YAML config and resolve environment variables."""
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Resolve environment variables
    resolved_content = _resolve_env_vars(content)
    
    return yaml.safe_load(resolved_content)


class DataConfig(BaseSettings):
    """Data pipeline configuration."""
    
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    validated_dir: str = "data/validated"
    
    # Storage
    endpoint_url: str = Field(default="http://minio:9000", env="AWS_ENDPOINT_URL")
    bucket: str = Field(default="industrial-qc-data", env="S3_BUCKET")
    access_key: str = Field(default="minioadmin", env="AWS_ACCESS_KEY_ID")
    secret_key: str = Field(default="minioadmin", env="AWS_SECRET_ACCESS_KEY")
    
    # Splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Image specs
    image_height: int = 640
    image_width: int = 640
    channels: int = 3
    
    # Classes
    classes: List[str] = ["crack", "inclusion", "stain", "pitting", "scale"]
    num_classes: int = 5
    
    # Validation
    min_images_per_class: int = 100
    max_imbalance_ratio: float = 5.0
    min_bbox_area: int = 16
    
    class Config:
        env_prefix = "DATA_"
        env_file = ".env"


class ModelConfig(BaseSettings):
    """Model architecture configuration."""
    
    variant: str = "s"  # n, s, m, l, x
    pretrained: bool = True
    pretrained_weights: str = "yolov8s.pt"
    
    num_classes: int = 5
    class_names: List[str] = ["crack", "inclusion", "stain", "pitting", "scale"]
    
    # Architecture
    depth_multiplier: float = 0.33
    width_multiplier: float = 0.5
    max_stride: int = 64
    
    # NMS
    iou_threshold: float = 0.45
    confidence_threshold: float = 0.25
    max_detections: int = 300
    
    # Quantization
    quantization_dtype: str = "fp16"  # fp16, int8
    
    class Config:
        env_prefix = "MODEL_"
        env_file = ".env"


class TrainingConfig(BaseSettings):
    """Training hyperparameters."""
    
    seed: int = 42
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    
    # Mixed precision
    amp: bool = True
    
    # Gradient clipping
    gradient_clip_val: float = 10.0
    
    # Optimizer
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # Scheduler
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5
    
    # Early stopping
    early_stopping_patience: int = 20
    
    # MLflow
    mlflow_experiment_name: str = Field(default="industrial-defect-detection", env="MLFLOW_EXPERIMENT_NAME")
    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", env="MLFLOW_TRACKING_URI")
    
    class Config:
        env_prefix = "TRAIN_"
        env_file = ".env"


class ServingConfig(BaseSettings):
    """TorchServe configuration."""
    
    port: int = 8080
    grpc_port: int = 7070
    num_workers: int = 2
    
    model_name: str = "defect_detector"
    model_version: str = "1.0.0"
    
    batch_size: int = 4
    max_batch_delay: int = 50  # ms
    
    # GPU
    gpu_enabled: bool = True
    memory_fraction: float = 0.8
    
    # Performance
    fp16_inference: bool = True
    warmup_requests: int = 10
    
    # Health checks
    liveness_endpoint: str = "/ping"
    readiness_endpoint: str = "/healthz"
    
    class Config:
        env_prefix = "SERVING_"
        env_file = ".env"


class ConfigManager:
    """
    Central configuration manager.
    
    Loads all configurations from YAML files and environment variables.
    Environment variables take precedence over file values.
    
    Usage:
        config = ConfigManager()
        data_config = config.data
        model_config = config.model
        
        # Access specific values
        print(config.data.bucket)
        print(config.training.learning_rate)
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager."""
        self.config_dir = config_dir or Path(__file__).parent.parent / "configs"
        
        # Load configs from files
        self._data_raw = self._load_config("data.yaml")
        self._model_raw = self._load_config("model.yaml")
        self._training_raw = self._load_config("training.yaml")
        self._serving_raw = self._load_config("serving.yaml")
        
        # Create Pydantic models
        self.data = self._create_data_config()
        self.model = self._create_model_config()
        self.training = self._create_training_config()
        self.serving = self._create_serving_config()
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        return _load_yaml_with_env_resolution(config_path)
    
    def _create_data_config(self) -> DataConfig:
        """Create DataConfig from loaded YAML."""
        data_section = self._data_raw.get("data", {})
        storage = self._data_raw.get("storage", {})
        splits = self._data_raw.get("splits", {})
        image = self._data_raw.get("image", {})
        validation = self._data_raw.get("validation", {})
        classes_cfg = self._data_raw.get("classes", [])
        
        return DataConfig(
            raw_dir=data_section.get("raw_dir", "data/raw"),
            processed_dir=data_section.get("processed_dir", "data/processed"),
            validated_dir=data_section.get("validated_dir", "data/validated"),
            endpoint_url=storage.get("endpoint_url", os.environ.get("AWS_ENDPOINT_URL", "http://minio:9000")),
            bucket=storage.get("bucket", os.environ.get("S3_BUCKET", "industrial-qc-data")),
            access_key=storage.get("access_key", os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")),
            secret_key=storage.get("secret_key", os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")),
            train_split=splits.get("train", 0.7),
            val_split=splits.get("val", 0.15),
            test_split=splits.get("test", 0.15),
            image_height=image.get("height", 640),
            image_width=image.get("width", 640),
            channels=image.get("channels", 3),
            classes=[c["name"] for c in classes_cfg],
            num_classes=len(classes_cfg) if classes_cfg else 5,
            min_images_per_class=validation.get("min_images_per_class", 100),
            max_imbalance_ratio=validation.get("max_imbalance_ratio", 5.0),
            min_bbox_area=validation.get("min_bbox_area", 16),
        )
    
    def _create_model_config(self) -> ModelConfig:
        """Create ModelConfig from loaded YAML."""
        model_section = self._model_raw.get("model", {})
        arch = self._model_raw.get("architecture", {})
        nms = self._model_raw.get("nms", {})
        quant = self._model_raw.get("quantization", {})
        
        return ModelConfig(
            variant=model_section.get("variant", "s"),
            pretrained=model_section.get("pretrained", True),
            pretrained_weights=model_section.get("pretrained_weights", "yolov8s.pt"),
            num_classes=model_section.get("num_classes", 5),
            class_names=model_section.get("class_names", ["crack", "inclusion", "stain", "pitting", "scale"]),
            depth_multiplier=arch.get("depth_multiplier", 0.33),
            width_multiplier=arch.get("width_multiplier", 0.5),
            max_stride=arch.get("max_stride", 64),
            iou_threshold=nms.get("iou_threshold", 0.45),
            confidence_threshold=nms.get("confidence_threshold", 0.25),
            max_detections=nms.get("max_detections", 300),
            quantization_dtype=quant.get("dtype", "fp16"),
        )
    
    def _create_training_config(self) -> TrainingConfig:
        """Create TrainingConfig from loaded YAML."""
        train_section = self._training_raw.get("training", {})
        optimizer = self._training_raw.get("optimizer", {})
        scheduler = self._training_raw.get("scheduler", {})
        early_stop = self._training_raw.get("early_stopping", {})
        mlflow_cfg = self._training_raw.get("mlflow", {})
        
        return TrainingConfig(
            seed=train_section.get("seed", 42),
            epochs=train_section.get("epochs", 100),
            batch_size=train_section.get("batch_size", 16),
            num_workers=train_section.get("num_workers", 4),
            amp=train_section.get("amp", True),
            gradient_clip_val=train_section.get("gradient_clip_val", 10.0),
            learning_rate=optimizer.get("lr", 0.01),
            momentum=optimizer.get("momentum", 0.937),
            weight_decay=optimizer.get("weight_decay", 0.0005),
            scheduler_type=scheduler.get("type", "cosine"),
            warmup_epochs=scheduler.get("warmup_epochs", 5),
            early_stopping_patience=early_stop.get("patience", 20),
            mlflow_experiment_name=mlflow_cfg.get("experiment_name", os.environ.get("MLFLOW_EXPERIMENT_NAME", "industrial-defect-detection")),
            mlflow_tracking_uri=mlflow_cfg.get("tracking_uri", os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")),
        )
    
    def _create_serving_config(self) -> ServingConfig:
        """Create ServingConfig from loaded YAML."""
        server = self._serving_raw.get("server", {})
        model = self._serving_raw.get("model", {})
        gpu = self._serving_raw.get("gpu", {})
        perf = self._serving_raw.get("performance", {})
        health = self._serving_raw.get("health", {})
        
        return ServingConfig(
            port=server.get("port", 8080),
            grpc_port=server.get("grpc_port", 7070),
            num_workers=server.get("num_workers", 2),
            model_name=model.get("name", "defect_detector"),
            model_version=model.get("version", "1.0.0"),
            batch_size=model.get("batch_size", 4),
            max_batch_delay=model.get("max_batch_delay", 50),
            gpu_enabled=gpu.get("enabled", True),
            memory_fraction=gpu.get("memory_fraction", 0.8),
            fp16_inference=perf.get("fp16", True),
            warmup_requests=perf.get("warmup_requests", 10),
            liveness_endpoint=health.get("liveness_endpoint", "/ping"),
            readiness_endpoint=health.get("readiness_endpoint", "/healthz"),
        )
    
    def validate(self) -> List[str]:
        """
        Validate configuration consistency.
        
        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        
        # Check splits sum to 1.0
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 0.01:
            errors.append(f"Data splits must sum to 1.0, got {total_split}")
        
        # Check num_classes consistency
        if self.data.num_classes != self.model.num_classes:
            errors.append(
                f"Class count mismatch: data has {self.data.num_classes}, "
                f"model expects {self.model.num_classes}"
            )
        
        # Check class names
        if set(self.data.classes) != set(self.model.class_names):
            errors.append("Class names mismatch between data and model configs")
        
        return errors


# Global config instance (lazy loaded)
_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config
