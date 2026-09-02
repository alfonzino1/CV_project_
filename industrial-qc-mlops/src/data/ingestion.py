"""Data ingestion from S3/MinIO with DVC integration and validation."""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from dvc.repo import Repo
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger(__name__)


class DatasetSchema(BaseModel):
    """Schema for dataset validation."""
    
    min_images: int = Field(default=100, ge=1)
    max_image_size_mb: float = Field(default=50.0, ge=1.0)
    required_extensions: List[str] = Field(default=[".jpg", ".jpeg", ".png"])
    min_annotations_per_class: int = Field(default=10, ge=1)
    train_split_ratio: float = Field(default=0.8, ge=0.5, le=0.9)
    val_split_ratio: float = Field(default=0.1, ge=0.05, le=0.3)
    test_split_ratio: float = Field(default=0.1, ge=0.05, le=0.3)
    
    class Config:
        frozen = True


class IngestionResult(BaseModel):
    """Result of data ingestion process."""
    
    success: bool
    total_images: int
    total_annotations: int
    classes: List[str]
    train_count: int
    val_count: int
    test_count: int
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DataIngestion:
    """
    Handle data download from S3/MinIO and DVC, with schema validation.
    
    This class manages:
    - Downloading raw data from S3-compatible storage
    - Pulling versioned datasets from DVC
    - Validating dataset structure and schema
    - Generating train/val/test splits
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        schema: Optional[DatasetSchema] = None,
    ) -> None:
        """
        Initialize DataIngestion.
        
        Args:
            settings: Application settings (uses global settings if None).
            schema: Dataset validation schema (uses default if None).
        """
        self.settings = settings or get_settings()
        self.schema = schema or DatasetSchema()
        
        # Setup paths
        self.raw_dir = Path(self.settings.data_raw_dir)
        self.processed_dir = Path(self.settings.data_processed_dir)
        self.splits_dir = Path(self.settings.data_splits_dir)
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client
        self.s3_client = self._create_s3_client()
        
        logger.info(
            "DataIngestion initialized",
            raw_dir=str(self.raw_dir),
            processed_dir=str(self.processed_dir),
            s3_endpoint=self.settings.minio_endpoint,
        )
    
    def _create_s3_client(self) -> boto3.client:
        """Create S3 client for MinIO/S3 interaction."""
        config = Config(
            retries={"max_attempts": 3, "mode": "standard"},
            signature_version="s3v4",
        )
        
        client = boto3.client(
            "s3",
            endpoint_url=self.settings.aws_endpoint_url,
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            config=config,
        )
        
        logger.debug("S3 client created", endpoint=self.settings.minio_endpoint)
        return client
    
    def download_from_s3(
        self,
        s3_prefix: str,
        destination: Optional[Path] = None,
    ) -> int:
        """
        Download data from S3 bucket.
        
        Args:
            s3_prefix: S3 prefix to download from.
            destination: Local destination path (uses raw_dir if None).
            
        Returns:
            Number of files downloaded.
        """
        dest = destination or self.raw_dir
        dest.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Downloading from S3",
            bucket=self.settings.minio_bucket,
            prefix=s3_prefix,
            destination=str(dest),
        )
        
        files_downloaded = 0
        
        try:
            # List objects with prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.settings.minio_bucket,
                Prefix=s3_prefix,
            )
            
            for page in pages:
                if "Contents" not in page:
                    continue
                    
                for obj in page["Contents"]:
                    key = obj["Key"]
                    
                    # Skip directories
                    if key.endswith("/"):
                        continue
                    
                    # Calculate relative path
                    relative_path = key[len(s3_prefix):].lstrip("/")
                    local_path = dest / relative_path
                    
                    # Create parent directories
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    logger.debug("Downloading file", key=key, local_path=str(local_path))
                    self.s3_client.download_file(
                        self.settings.minio_bucket,
                        key,
                        str(local_path),
                    )
                    files_downloaded += 1
            
            logger.info(
                "Download completed",
                files_downloaded=files_downloaded,
            )
            
        except Exception as e:
            logger.error("S3 download failed", error=str(e), exc_info=True)
            raise
        
        return files_downloaded
    
    def pull_from_dvc(self) -> int:
        """
        Pull versioned data from DVC remote.
        
        Returns:
            Number of files pulled.
        """
        logger.info("Pulling data from DVC remote", remote=self.settings.dvc_remote)
        
        try:
            repo = Repo()
            repo.pull()
            
            # Count pulled files
            files_pulled = sum(
                1 for _ in self.raw_dir.rglob("*") 
                if _.is_file() and not _.name.startswith(".")
            )
            
            logger.info("DVC pull completed", files_pulled=files_pulled)
            return files_pulled
            
        except Exception as e:
            logger.error("DVC pull failed", error=str(e), exc_info=True)
            # DVC might not be initialized yet, which is OK for first run
            logger.warning("DVC not initialized, skipping pull")
            return 0
    
    def validate_schema(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate dataset against schema.
        
        Returns:
            Tuple of (is_valid, errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        logger.info("Validating dataset schema")
        
        # Find all images
        images: List[Path] = []
        for ext in self.schema.required_extensions:
            images.extend(self.raw_dir.rglob(f"*{ext}"))
            images.extend(self.raw_dir.rglob(f"*{ext.upper()}"))
        
        # Check minimum images
        if len(images) < self.schema.min_images:
            errors.append(
                f"Insufficient images: {len(images)} < {self.schema.min_images}"
            )
        
        # Check image sizes
        for img_path in images:
            size_mb = img_path.stat().st_size / (1024 * 1024)
            if size_mb > self.schema.max_image_size_mb:
                warnings.append(
                    f"Large image detected: {img_path.name} ({size_mb:.2f} MB)"
                )
        
        # Check annotations (YOLO format: .txt files)
        annotations = list(self.raw_dir.rglob("*.txt"))
        if not annotations:
            warnings.append("No annotation files found (.txt)")
        
        # Check class distribution
        class_counts: Dict[str, int] = {}
        for ann_file in annotations:
            try:
                with open(ann_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_name = (
                                self.settings.defect_classes[class_id]
                                if class_id < len(self.settings.defect_classes)
                                else f"class_{class_id}"
                            )
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            except Exception:
                continue
        
        # Check minimum annotations per class
        for class_name in self.settings.defect_classes:
            count = class_counts.get(class_name, 0)
            if count < self.schema.min_annotations_per_class:
                warnings.append(
                    f"Class '{class_name}' has only {count} annotations "
                    f"(min: {self.schema.min_annotations_per_class})"
                )
        
        # Check split ratios
        total_ratio = (
            self.schema.train_split_ratio +
            self.schema.val_split_ratio +
            self.schema.test_split_ratio
        )
        if abs(total_ratio - 1.0) > 0.01:
            errors.append(
                f"Split ratios must sum to 1.0, got {total_ratio:.2f}"
            )
        
        is_valid = len(errors) == 0
        
        logger.info(
            "Schema validation completed",
            is_valid=is_valid,
            num_images=len(images),
            num_annotations=len(annotations),
            num_errors=len(errors),
            num_warnings=len(warnings),
        )
        
        return is_valid, errors, warnings
    
    def generate_splits(
        self,
        seed: int = 42,
    ) -> Dict[str, List[Path]]:
        """
        Generate train/val/test splits.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with 'train', 'val', 'test' lists of image paths.
        """
        import random
        
        logger.info("Generating dataset splits", seed=seed)
        
        # Find all images
        images: List[Path] = []
        for ext in self.schema.required_extensions:
            images.extend(self.raw_dir.rglob(f"*{ext}"))
            images.extend(self.raw_dir.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort for consistency
        images = sorted(list(set(images)))
        random.seed(seed)
        random.shuffle(images)
        
        # Calculate split sizes
        n_total = len(images)
        n_train = int(n_total * self.schema.train_split_ratio)
        n_val = int(n_total * self.schema.val_split_ratio)
        n_test = n_total - n_train - n_val
        
        # Split
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Save split manifests
        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }
        
        for split_name, split_images in splits.items():
            manifest_path = self.splits_dir / f"{split_name}.txt"
            with open(manifest_path, "w") as f:
                for img_path in split_images:
                    # Write relative path
                    rel_path = img_path.relative_to(self.raw_dir)
                    f.write(f"{rel_path}\n")
            
            logger.info(
                f"{split_name} split saved",
                num_images=len(split_images),
                manifest=str(manifest_path),
            )
        
        return splits
    
    def run(
        self,
        s3_prefix: Optional[str] = None,
        use_dvc: bool = True,
        validate: bool = True,
        generate_splits_flag: bool = True,
    ) -> IngestionResult:
        """
        Run complete ingestion pipeline.
        
        Args:
            s3_prefix: S3 prefix to download from (skips if None).
            use_dvc: Whether to pull from DVC.
            validate: Whether to validate schema.
            generate_splits_flag: Whether to generate splits.
            
        Returns:
            IngestionResult with statistics and errors.
        """
        logger.info("Starting data ingestion pipeline")
        
        errors: List[str] = []
        warnings: List[str] = []
        
        # Step 1: Download from S3
        if s3_prefix:
            try:
                self.download_from_s3(s3_prefix)
            except Exception as e:
                errors.append(f"S3 download failed: {str(e)}")
        
        # Step 2: Pull from DVC
        if use_dvc:
            try:
                self.pull_from_dvc()
            except Exception as e:
                warnings.append(f"DVC pull skipped: {str(e)}")
        
        # Step 3: Validate schema
        if validate:
            is_valid, schema_errors, schema_warnings = self.validate_schema()
            errors.extend(schema_errors)
            warnings.extend(schema_warnings)
        
        # Step 4: Generate splits
        splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
        if generate_splits_flag:
            try:
                splits = self.generate_splits()
            except Exception as e:
                errors.append(f"Split generation failed: {str(e)}")
        
        # Count results
        total_images = sum(len(imgs) for imgs in splits.values())
        total_annotations = len(list(self.raw_dir.rglob("*.txt")))
        
        result = IngestionResult(
            success=len(errors) == 0,
            total_images=total_images,
            total_annotations=total_annotations,
            classes=self.settings.defect_classes,
            train_count=len(splits["train"]),
            val_count=len(splits["val"]),
            test_count=len(splits["test"]),
            errors=errors,
            warnings=warnings,
        )
        
        logger.info(
            "Data ingestion completed",
            success=result.success,
            total_images=result.total_images,
            train_count=result.train_count,
            val_count=result.val_count,
            test_count=result.test_count,
        )
        
        return result
