"""Evaluation metrics for defect detection models."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluate trained defect detection models.

    Metrics:
    - mAP@50, mAP@50-95
    - Precision, Recall, F1-score
    - Per-class metrics
    - Confusion matrix
    - Inference speed
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        self.models_dir = Path(self.settings.models_dir)

        logger.info("ModelEvaluator initialized")

    def evaluate(
        self,
        model_path: str,
        data_yaml: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Run full evaluation.

        Args:
            model_path: Path to model weights.
            data_yaml: Path to dataset YAML.
            conf_threshold: Confidence threshold.
            iou_threshold: IoU threshold for NMS.

        Returns:
            Dictionary with all evaluation metrics.
        """
        logger.info("Starting evaluation", model_path=model_path)

        # Load model
        model = YOLO(model_path)

        # Run validation
        results = model.val(
            data=str(data_yaml),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        # Extract metrics
        metrics = {
            # Overall metrics
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "F1": float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-6)),

            # Per-class metrics
            "per_class_mAP50": results.box.maps.tolist(),
            "per_class_mAP50-95": results.box.map50_95s.tolist() if hasattr(results.box, 'map50_95s') else [],

            # Losses
            "box_loss": float(results.box_loss) if hasattr(results, 'box_loss') else 0.0,
            "cls_loss": float(results.cls_loss) if hasattr(results, 'cls_loss') else 0.0,
            "dfl_loss": float(results.dfl_loss) if hasattr(results, 'dfl_loss') else 0.0,

            # Class names
            "class_names": list(model.names.values()) if hasattr(model, 'names') else [],
        }

        # Calculate per-class F1
        if hasattr(results, 'box') and hasattr(results.box, 'p') and hasattr(results.box, 'r'):
            per_class_precision = results.box.p.tolist() if hasattr(results.box, 'p') else []
            per_class_recall = results.box.r.tolist() if hasattr(results.box, 'r') else []

            per_class_f1 = []
            for p, r in zip(per_class_precision, per_class_recall):
                f1 = 2 * p * r / (p + r + 1e-6)
                per_class_f1.append(float(f1))

            metrics["per_class_F1"] = per_class_f1

        logger.info("Evaluation completed", mAP50_95=metrics["mAP50-95"])

        return metrics

    def compute_confusion_matrix(
        self,
        model_path: str,
        data_yaml: Path,
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            model_path: Path to model weights.
            data_yaml: Path to dataset YAML.

        Returns:
            Confusion matrix array.
        """
        logger.info("Computing confusion matrix", model_path=model_path)

        model = YOLO(model_path)

        # Run validation with confusion matrix computation
        results = model.val(
            data=str(data_yaml),
            verbose=False,
        )

        # Extract confusion matrix if available
        if hasattr(results, 'confusion_matrix'):
            cm = results.confusion_matrix.matrix
            logger.info("Confusion matrix computed", shape=cm.shape)
            return cm
        else:
            logger.warning("Confusion matrix not available")
            return np.array([])

    def analyze_errors(
        self,
        model_path: str,
        data_yaml: Path,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors.

        Args:
            model_path: Path to model weights.
            data_yaml: Path to dataset YAML.
            output_dir: Directory to save error analysis.

        Returns:
            Error analysis report.
        """
        output_dir = output_dir or self.models_dir / "error_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Analyzing errors", model_path=model_path)

        model = YOLO(model_path)

        # Run validation and collect predictions
        results = model.val(
            data=str(data_yaml),
            verbose=False,
        )

        # Collect false positives and false negatives
        error_report = {
            "false_positives": [],
            "false_negatives": [],
            "low_confidence_detections": [],
            "small_objects_missed": [],
        }

        # This would require accessing individual predictions vs ground truth
        # For now, we'll use aggregate statistics
        error_report["summary"] = {
            "total_false_positives": int((1 - results.box.mp) * results.num_samples),
            "total_false_negatives": int((1 - results.box.mr) * results.num_samples),
            "average_confidence": float(np.mean([r.boxes.conf.max() if len(r.boxes) > 0 else 0 for r in results.pred])) if hasattr(results, 'pred') else 0.0,
        }

        # Save report
        report_path = output_dir / "error_report.json"
        with open(report_path, "w") as f:
            json.dump(error_report, f, indent=2)

        logger.info("Error analysis saved", path=str(report_path))

        return error_report

    def log_to_mlflow(
        self,
        metrics: Dict[str, Any],
        run_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Log evaluation metrics to MLflow.

        Args:
            metrics: Evaluation metrics dictionary.
            run_id: MLflow run ID.
            experiment_name: Experiment name.
        """
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)

        if experiment_name:
            mlflow.set_experiment(experiment_name)

        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run()

        try:
            # Log main metrics
            mlflow.log_metric("mAP50", metrics["mAP50"])
            mlflow.log_metric("mAP50-95", metrics["mAP50-95"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("F1", metrics["F1"])

            # Log per-class metrics
            for i, class_name in enumerate(metrics.get("class_names", [])):
                if i < len(metrics.get("per_class_mAP50", [])):
                    mlflow.log_metric(f"{class_name}_mAP50", metrics["per_class_mAP50"][i])
                if i < len(metrics.get("per_class_F1", [])):
                    mlflow.log_metric(f"{class_name}_F1", metrics["per_class_F1"][i])

            # Log losses
            mlflow.log_metric("box_loss", metrics.get("box_loss", 0.0))
            mlflow.log_metric("cls_loss", metrics.get("cls_loss", 0.0))

            logger.info("Metrics logged to MLflow", run_id=mlflow.active_run().info.run_id)

        finally:
            mlflow.end_run()

    def generate_report(
        self,
        metrics: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """
        Generate evaluation report.

        Args:
            metrics: Evaluation metrics.
            output_path: Path to save report.

        Returns:
            Path to generated report.
        """
        import yaml

        report = {
            "evaluation_summary": {
                "mAP50": metrics["mAP50"],
                "mAP50-95": metrics["mAP50-95"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "F1_score": metrics["F1"],
            },
            "per_class_metrics": {},
            "model_info": {
                "classes": metrics.get("class_names", []),
                "num_classes": len(metrics.get("class_names", [])),
            },
        }

        # Add per-class metrics
        class_names = metrics.get("class_names", [])
        for i, class_name in enumerate(class_names):
            report["per_class_metrics"][class_name] = {
                "mAP50": metrics["per_class_mAP50"][i] if i < len(metrics.get("per_class_mAP50", [])) else None,
                "F1": metrics["per_class_F1"][i] if i < len(metrics.get("per_class_F1", [])) else None,
            }

        # Save report
        with open(output_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)

        logger.info("Report generated", path=str(output_path))

        return output_path


def evaluate_model(
    model_path: str,
    data_yaml: Path,
    output_dir: Optional[Path] = None,
    log_mlflow: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for model evaluation.

    Args:
        model_path: Path to model weights.
        data_yaml: Path to dataset YAML.
        output_dir: Output directory for reports.
        log_mlflow: Whether to log to MLflow.

    Returns:
        Evaluation metrics dictionary.
    """
    settings = get_settings()
    evaluator = ModelEvaluator(settings=settings)

    output_dir = output_dir or Path(settings.models_dir) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    metrics = evaluator.evaluate(model_path, data_yaml)

    # Generate report
    report_path = output_dir / "evaluation_report.yaml"
    evaluator.generate_report(metrics, report_path)

    # Log to MLflow
    if log_mlflow:
        evaluator.log_to_mlflow(metrics, experiment_name=settings.mlflow_experiment_name)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Dataset YAML")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    metrics = evaluate_model(
        model_path=args.model,
        data_yaml=Path(args.data),
        output_dir=Path(args.output) if args.output else None,
    )

    print("\n=== Evaluation Results ===")
    print(f"mAP50: {metrics['mAP50']:.4f}")
    print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['F1']:.4f}")
