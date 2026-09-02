"""Configuration management using Pydantic."""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # === S3/MinIO Configuration ===
    minio_endpoint: str = Field(default="http://localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="industrial-qc-data", env="MINIO_BUCKET")
    
    # === MLflow Configuration ===
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="industrial-qc-defect-detection", 
        env="MLFLOW_EXPERIMENT_NAME"
    )
    
    # === DVC Configuration ===
    dvc_remote: str = Field(default="s3://industrial-qc-data/dvc", env="DVC_REMOTE")
    dvc_remote_region: str = Field(default="us-east-1", env="DVC_REMOTE_REGION")
    
    # === Training Configuration ===
    gpu_ids: str = Field(default="0", env="GPU_IDS")
    batch_size: int = Field(default=16, env="BATCH_SIZE", ge=1, le=128)
    epochs: int = Field(default=100, env="EPOCHS", ge=1, le=1000)
    image_size: int = Field(default=640, env="IMAGE_SIZE", ge=320, le=1280)
    num_classes: int = Field(default=5, env="NUM_CLASSES", ge=1, le=100)
    model_name: str = Field(default="yolov8n", env="MODEL_NAME")
    
    # === Serving Configuration ===
    torchserve_port: int = Field(default=8080, env="TORCHSERVE_PORT")
    max_batch_size: int = Field(default=8, env="MAX_BATCH_SIZE", ge=1, le=32)
    inference_timeout: int = Field(default=50, env="INFERENCE_TIMEOUT", ge=10, le=1000)
    quantization_mode: str = Field(default="fp16", env="QUANTIZATION_MODE")
    
    # === Monitoring Configuration ===
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    drift_threshold: float = Field(default=0.1, env="DRIFT_THRESHOLD", ge=0.01, le=1.0)
    
    # === Kubernetes Configuration ===
    k8s_namespace: str = Field(default="industrial-qc", env="K8S_NAMESPACE")
    k8s_replicas: int = Field(default=3, env="K8S_REPLICAS", ge=1, le=10)
    k8s_gpu_enabled: bool = Field(default=True, env="K8S_GPU_ENABLED")
    
    # === Prefect Configuration ===
    prefect_api_url: str = Field(default="http://localhost:4200/api", env="PREFECT_API_URL")
    prefect_agent_name: str = Field(default="industrial-qc-agent", env="PREFECT_AGENT_NAME")
    
    # === Application Settings ===
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # === Data Paths ===
    data_raw_dir: str = Field(default="data/raw", env="DATA_RAW_DIR")
    data_processed_dir: str = Field(default="data/processed", env="DATA_PROCESSED_DIR")
    data_splits_dir: str = Field(default="data/splits", env="DATA_SPLITS_DIR")
    models_dir: str = Field(default="models", env="MODELS_DIR")
    
    # === Defect Classes (Industrial QC) ===
    defect_classes: List[str] = Field(
        default=["crack", "inclusion", "stain", "pitting", "scale"],
        env="DEFECT_CLASSES"
    )
    
    @validator("model_name")
    def validate_model_name(cls, v: str) -> str:
        """Validate YOLOv8 model name."""
        valid_models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        if v not in valid_models:
            raise ValueError(
                f"Invalid model_name: {v}. Must be one of {valid_models}"
            )
        return v
    
    @validator("quantization_mode")
    def validate_quantization_mode(cls, v: str) -> str:
        """Validate quantization mode."""
        valid_modes = ["fp16", "int8", "none"]
        if v.lower() not in valid_modes:
            raise ValueError(
                f"Invalid quantization_mode: {v}. Must be one of {valid_modes}"
            )
        return v.lower()
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log_level: {v}. Must be one of {valid_levels}"
            )
        return v.upper()
    
    @property
    def aws_endpoint_url(self) -> str:
        """Get AWS endpoint URL for boto3."""
        return self.minio_endpoint
    
    @property
    def aws_access_key_id(self) -> str:
        """Get AWS access key ID."""
        return self.minio_access_key
    
    @property
    def aws_secret_access_key(self) -> str:
        """Get AWS secret access key."""
        return self.minio_secret_key
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def device(self) -> str:
        """Get compute device (cuda or cpu)."""
        if self.gpu_ids and self.gpu_ids != "-1":
            return f"cuda:{self.gpu_ids.split(',')[0]}"
        return "cpu"
    
    class Config:
        """Pydantic config."""
        
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance (for dependency injection)."""
    return settings
