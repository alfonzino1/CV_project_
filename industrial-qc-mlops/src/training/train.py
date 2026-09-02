"""YOLOv8 training with PyTorch Lightning and MLflow integration."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import torch
import yaml
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger(__name__)


class DefectDetectionDataset(Dataset):
    """PyTorch Dataset for YOLO format defect detection."""

    def __init__(
        self,
        images_dir: Path,
        annotations_dir: Path,
        image_size: int = 640,
        augment: bool = False,
    ) -> None:
        """
        Initialize dataset.

        Args:
            images_dir: Directory containing images.
            annotations_dir: Directory containing YOLO annotations.
            image_size: Target image size.
            augment: Whether to apply augmentations.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_size = image_size
        self.augment = augment

        # Get all image files
        self.image_paths = list(images_dir.glob("*.jpg")) + \
                          list(images_dir.glob("*.jpeg")) + \
                          list(images_dir.glob("*.png"))

        logger.info(
            "Dataset initialized",
            num_images=len(self.image_paths),
            augment=augment,
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with image tensor and targets.
        """
        from PIL import Image
        import numpy as np
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # Load annotations
        ann_path = self.annotations_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if ann_path.exists():
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append([x_center, y_center, width, height])
                        labels.append(class_id)

        # Apply augmentations
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5 if self.augment else 0.0),
                A.RandomBrightnessContrast(p=0.3 if self.augment else 0.0),
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
        )

        transformed = transform(
            image=img_array,
            bboxes=boxes,
            labels=labels,
        )

        return {
            "image": transformed["image"],
            "boxes": torch.tensor(transformed["bboxes"], dtype=torch.float32) if transformed["bboxes"] else torch.zeros((0, 4)),
            "labels": torch.tensor(transformed["class_labels"], dtype=torch.int64),
            "image_path": str(img_path),
        }


class YOLOv8LightningModule(LightningModule):
    """PyTorch Lightning module for YOLOv8 training."""

    def __init__(
        self,
        model_name: str = "yolov8n",
        num_classes: int = 5,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0005,
        imgsz: int = 640,
    ) -> None:
        """
        Initialize Lightning module.

        Args:
            model_name: YOLOv8 model variant.
            num_classes: Number of defect classes.
            learning_rate: Initial learning rate.
            weight_decay: Weight decay for optimizer.
            imgsz: Image size for training.
        """
        super().__init__()
        self.save_hyperparameters()

        # Load YOLOv8 model
        self.model = YOLO(f"{model_name}.pt")

        # Override number of classes
        if num_classes != 80:  # COCO has 80 classes
            self.model.model[-1].nc = num_classes
            self.model.model[-1].shape = (num_classes,)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.imgsz = imgsz
        self.num_classes = num_classes

        # Metrics will be computed by YOLO during validation
        self.val_metrics: Dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass."""
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.937,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=self.learning_rate * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch["image"]
        targets = {
            "boxes": batch["boxes"],
            "labels": batch["labels"],
        }

        # YOLOv8 training returns loss directly
        loss = self.model.train_step(batch, batch_idx)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Validation step."""
        images = batch["image"]
        targets = {
            "boxes": batch["boxes"],
            "labels": batch["labels"],
        }

        # Run inference
        predictions = self.model.predict(images, verbose=False)

        # Compute metrics (mAP, etc.)
        # Note: Full mAP computation requires all predictions at once
        # We'll aggregate in on_validation_epoch_end

        return {"predictions": predictions, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        # Metrics are computed by YOLO's built-in validator
        # Access via self.model.metrics
        if hasattr(self.model, "metrics"):
            metrics = self.model.metrics

            self.log("val_mAP50", metrics.get("metrics/mAP50", 0.0), sync_dist=True)
            self.log("val_mAP50-95", metrics.get("metrics/mAP50-95(B)", 0.0), sync_dist=True)
            self.log("val_precision", metrics.get("metrics/precision(B)", 0.0), sync_dist=True)
            self.log("val_recall", metrics.get("metrics/recall(B)", 0.0), sync_dist=True)


class YOLOv8Trainer:
    """
    High-level trainer for YOLOv8 defect detection models.

    Features:
    - MLflow experiment tracking
    - PyTorch Lightning training loop
    - Automatic checkpointing
    - Early stopping
    - Hyperparameter logging
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            settings: Application settings.
            config_path: Path to training configuration YAML.
        """
        self.settings = settings or get_settings()

        # Load config
        self.config = {}
        if config_path and config_path.exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        # Setup output directory
        self.output_dir = Path(self.config.get("output_dir", "models/experiments"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "YOLOv8Trainer initialized",
            output_dir=str(self.output_dir),
            model=self.settings.model_name,
        )

    def train(
        self,
        data_yaml: Path,
        model_name: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        imgsz: Optional[int] = None,
        resume: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train YOLOv8 model.

        Args:
            data_yaml: Path to dataset YAML configuration.
            model_name: Model variant (overrides settings).
            epochs: Number of epochs (overrides config).
            batch_size: Batch size (overrides config).
            imgsz: Image size (overrides config).
            resume: Path to checkpoint to resume from.
            run_name: MLflow run name.

        Returns:
            Dictionary with training results and metrics.
        """
        model_name = model_name or self.settings.model_name
        epochs = epochs or self.config.get("epochs", 100)
        batch_size = batch_size or self.settings.batch_size
        imgsz = imgsz or self.settings.image_size

        # Generate run name
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{model_name}-{timestamp}"

        # Setup MLflow
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

        logger.info(
            "Starting training",
            model=model_name,
            epochs=epochs,
            batch_size=batch_size,
            image_size=imgsz,
            run_name=run_name,
        )

        try:
            with mlflow.start_run(run_name=run_name) as run:
                # Log hyperparameters
                mlflow.log_params({
                    "model": model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "imgsz": imgsz,
                    "num_classes": self.settings.num_classes,
                    "learning_rate": self.config.get("lr0", 0.01),
                    "weight_decay": self.config.get("weight_decay", 0.0005),
                })

                # Load model
                if resume:
                    model = YOLO(resume)
                    logger.info("Resuming from checkpoint", path=resume)
                else:
                    model = YOLO(f"{model_name}.pt")
                    logger.info("Loaded pretrained model", model=model_name)

                # Train
                results = model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=imgsz,
                    device=int(self.settings.gpu_ids.split(",")[0]) if self.settings.gpu_ids != "-1" else "cpu",
                    workers=self.config.get("workers", 8),
                    optimizer=self.config.get("optimizer", "SGD"),
                    lr0=self.config.get("lr0", 0.01),
                    lrf=self.config.get("lrf", 0.01),
                    momentum=self.config.get("momentum", 0.937),
                    weight_decay=self.config.get("weight_decay", 0.0005),
                    warmup_epochs=self.config.get("warmup_epochs", 3),
                    patience=self.config.get("patience", 50),
                    save=True,
                    save_period=self.config.get("checkpoint_interval", 10),
                    project=str(self.output_dir),
                    name=run_name,
                    exist_ok=True,
                    amp=self.config.get("amp", True),
                    cos_lr=self.config.get("cos_lr", True),
                    close_mosaic=self.config.get("close_mosaic", 10),
                    seed=self.config.get("seed", 42),
                )

                # Log final metrics
                final_metrics = {
                    "mAP50": float(results.results_dict.get("metrics/mAP50(val)", 0.0)),
                    "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)(val)", 0.0)),
                    "precision": float(results.results_dict.get("metrics/precision(B)(val)", 0.0)),
                    "recall": float(results.results_dict.get("metrics/recall(B)(val)", 0.0)),
                    "box_loss": float(results.results_dict.get("train/box_loss", 0.0)),
                    "cls_loss": float(results.results_dict.get("train/cls_loss", 0.0)),
                }

                mlflow.log_metrics(final_metrics)

                # Log model artifact
                best_model_path = self.output_dir / run_name / "weights" / "best.pt"
                if best_model_path.exists():
                    mlflow.pytorch.log_model(
                        model.model,
                        "model",
                        registered_model_name=f"industrial-qc-{model_name}",
                    )

                logger.info(
                    "Training completed",
                    run_id=run.info.run_id,
                    metrics=final_metrics,
                )

                return {
                    "success": True,
                    "run_id": run.info.run_id,
                    "model_path": str(best_model_path),
                    "metrics": final_metrics,
                }

        except Exception as e:
            logger.error("Training failed", error=str(e), exc_info=True)
            mlflow.log_param("error", str(e))
            return {
                "success": False,
                "error": str(e),
            }

    def validate(
        self,
        model_path: str,
        data_yaml: Path,
    ) -> Dict[str, float]:
        """
        Validate trained model.

        Args:
            model_path: Path to model weights.
            data_yaml: Path to dataset YAML.

        Returns:
            Dictionary with validation metrics.
        """
        logger.info("Starting validation", model_path=model_path)

        model = YOLO(model_path)

        results = model.val(
            data=str(data_yaml),
            device=int(self.settings.gpu_ids.split(",")[0]) if self.settings.gpu_ids != "-1" else "cpu",
        )

        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }

        logger.info("Validation completed", metrics=metrics)

        return metrics
