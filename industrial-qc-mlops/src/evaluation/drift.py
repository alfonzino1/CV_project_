"""Data drift detection for images using Evidently AI."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)
from evidently.metrics.data_drift.embedding_drift import EmbeddingDriftMetric
from PIL import Image
import torch
from torchvision import models, transforms

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger(__name__)


class ImageDriftDetector:
    """
    Detect data drift in image datasets.

    Features:
    - Visual feature drift (lighting, weather, angles)
    - Embedding-based drift detection
    - PSI (Population Stability Index) calculation
    - Automated alerting
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize drift detector.

        Args:
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        self.drift_threshold = self.settings.drift_threshold
        
        # Load pre-trained model for feature extraction
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.eval()
        
        # Image preprocessing for feature extraction
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info("ImageDriftDetector initialized", threshold=self.drift_threshold)

    def extract_features(self, image_paths: List[Path]) -> np.ndarray:
        """
        Extract features from images using ResNet.

        Args:
            image_paths: List of image paths.

        Returns:
            Feature matrix (num_images x feature_dim).
        """
        features = []

        with torch.no_grad():
            for img_path in image_paths:
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.preprocess(img).unsqueeze(0)

                    # Extract features
                    feat = self.feature_extractor(img_tensor)
                    features.append(feat.numpy().flatten())

                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
                    continue

        return np.array(features)

    def compute_psi(
        self,
        reference_values: np.ndarray,
        current_values: np.ndarray,
        buckets: int = 10,
    ) -> float:
        """
        Compute Population Stability Index.

        Args:
            reference_values: Reference distribution values.
            current_values: Current distribution values.
            buckets: Number of buckets for histogram.

        Returns:
            PSI value.
        """
        # Create bins based on reference distribution
        bins = np.linspace(reference_values.min(), reference_values.max(), buckets + 1)

        # Calculate percentages
        ref_hist, _ = np.histogram(reference_values, bins=bins)
        cur_hist, _ = np.histogram(current_values, bins=bins)

        # Convert to percentages
        ref_pct = (ref_hist + 1) / len(reference_values)  # Add 1 to avoid division by zero
        cur_pct = (cur_hist + 1) / len(current_values)

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    def detect_drift(
        self,
        reference_image_paths: List[Path],
        current_image_paths: List[Path],
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current datasets.

        Args:
            reference_image_paths: Paths to reference images.
            current_image_paths: Paths to current images.

        Returns:
            Drift detection results.
        """
        logger.info(
            "Detecting drift",
            reference_count=len(reference_image_paths),
            current_count=len(current_image_paths),
        )

        # Extract features
        ref_features = self.extract_features(reference_image_paths)
        cur_features = self.extract_features(current_image_paths)

        if len(ref_features) == 0 or len(cur_features) == 0:
            return {
                "drift_detected": False,
                "error": "Failed to extract features",
            }

        # Compute drift metrics per feature dimension
        psi_values = []
        for i in range(min(ref_features.shape[1], cur_features.shape[1])):
            psi = self.compute_psi(ref_features[:, i], cur_features[:, i])
            psi_values.append(psi)

        avg_psi = float(np.mean(psi_values))
        max_psi = float(np.max(psi_values))

        # Determine if drift is detected
        drift_detected = avg_psi > self.drift_threshold

        results = {
            "drift_detected": drift_detected,
            "avg_psi": avg_psi,
            "max_psi": max_psi,
            "threshold": self.drift_threshold,
            "num_features_analyzed": len(psi_values),
            "reference_samples": len(ref_features),
            "current_samples": len(cur_features),
        }

        logger.info(
            "Drift detection completed",
            drift_detected=drift_detected,
            avg_psi=avg_psi,
        )

        return results

    def generate_evidently_report(
        self,
        reference_image_paths: List[Path],
        current_image_paths: List[Path],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate detailed drift report using Evidently AI.

        Args:
            reference_image_paths: Paths to reference images.
            current_image_paths: Paths to current images.
            output_path: Output path for HTML report.

        Returns:
            Path to generated report.
        """
        logger.info("Generating Evidently report")

        # Extract features
        ref_features = self.extract_features(reference_image_paths)
        cur_features = self.extract_features(current_image_paths)

        # Convert to DataFrame
        ref_df = pd.DataFrame(ref_features, columns=[f"feature_{i}" for i in range(ref_features.shape[1])])
        cur_df = pd.DataFrame(cur_features, columns=[f"feature_{i}" for i in range(cur_features.shape[1])])

        # Create Evidently report
        report = Report(metrics=[
            DatasetDriftMetric(drift_threshold=self.drift_threshold),
            DataDriftTable(all_columns=False),
            EmbeddingDriftMetric(),
        ])

        report.run(reference_data=ref_df, current_data=cur_df)

        # Save report
        if output_path is None:
            output_path = Path(self.settings.models_dir) / "drift_report.html"
        else:
            output_path = Path(output_path)

        report.save_html(str(output_path))

        logger.info("Evidently report saved", path=str(output_path))

        return output_path

    def check_and_alert(
        self,
        reference_image_paths: List[Path],
        current_image_paths: List[Path],
        alert_callback: Optional[callable] = None,
    ) -> bool:
        """
        Check for drift and send alert if detected.

        Args:
            reference_image_paths: Paths to reference images.
            current_image_paths: Paths to current images.
            alert_callback: Callback function for alerts.

        Returns:
            True if drift detected, False otherwise.
        """
        results = self.detect_drift(reference_image_paths, current_image_paths)

        if results["drift_detected"]:
            logger.warning(
                "DATA DRIFT DETECTED!",
                avg_psi=results["avg_psi"],
                max_psi=results["max_psi"],
            )

            # Send alert
            if alert_callback:
                alert_callback(results)

            return True

        return False


def detect_data_drift(
    reference_dir: Path,
    current_dir: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function for drift detection.

    Args:
        reference_dir: Directory with reference images.
        current_dir: Directory with current images.
        output_dir: Output directory for reports.

    Returns:
        Drift detection results.
    """
    settings = get_settings()
    detector = ImageDriftDetector(settings=settings)

    # Collect image paths
    ref_images = list(reference_dir.glob("*.jpg")) + \
                 list(reference_dir.glob("*.png"))
    cur_images = list(current_dir.glob("*.jpg")) + \
                 list(current_dir.glob("*.png"))

    if not ref_images or not cur_images:
        return {
            "drift_detected": False,
            "error": "No images found",
        }

    # Detect drift
    results = detector.detect_drift(ref_images, cur_images)

    # Generate report
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "drift_report.html"
        detector.generate_evidently_report(ref_images, cur_images, report_path)
        results["report_path"] = str(report_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect data drift")
    parser.add_argument("--reference", type=str, required=True, help="Reference image directory")
    parser.add_argument("--current", type=str, required=True, help="Current image directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    results = detect_data_drift(
        reference_dir=Path(args.reference),
        current_dir=Path(args.current),
        output_dir=Path(args.output) if args.output else None,
    )

    print("\n=== Drift Detection Results ===")
    print(f"Drift Detected: {results['drift_detected']}")
    print(f"Average PSI: {results.get('avg_psi', 'N/A'):.4f}" if 'avg_psi' in results else "N/A")
    print(f"Max PSI: {results.get('max_psi', 'N/A'):.4f}" if 'max_psi' in results else "N/A")
