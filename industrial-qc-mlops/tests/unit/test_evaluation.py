"""Unit tests for model evaluator."""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.evaluation.metrics import ModelEvaluator


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create ModelEvaluator instance."""
        return ModelEvaluator(class_names=["crack", "inclusion", "stain", "pitting", "scale"])

    def test_calculate_metrics_basic(self, evaluator):
        """Test basic metrics calculation."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert "mAP_50" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_calculate_metrics_with_confidence(self, evaluator):
        """Test metrics with confidence scores."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 1])  # One misclassification
        confidences = np.array([0.9, 0.8, 0.7])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, confidences)
        
        assert metrics["mAP_50"] >= 0.0
        assert metrics["mAP_50"] <= 1.0

    def test_confusion_matrix_generation(self, evaluator, tmp_path):
        """Test confusion matrix generation and saving."""
        y_true = np.random.randint(0, 5, size=50)
        y_pred = np.random.randint(0, 5, size=50)
        
        cm_path = tmp_path / "confusion_matrix.png"
        result_path = evaluator.plot_confusion_matrix(y_true, y_pred, cm_path)
        
        assert result_path.exists()
        assert result_path.suffix == ".png"

    def test_per_class_metrics(self, evaluator):
        """Test per-class metrics calculation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert "per_class_metrics" in metrics or len(metrics.get("per_class_mAP50", [])) > 0

    @patch('src.evaluation.metrics.mlflow')
    def test_log_to_mlflow(self, mock_mlflow, evaluator, tmp_path):
        """Test MLflow logging."""
        metrics = {
            "mAP_50": 0.85,
            "precision": 0.88,
            "recall": 0.82,
            "f1_score": 0.85,
            "class_names": ["crack", "inclusion"]
        }
        
        # Create dummy artifact
        artifact = tmp_path / "test.txt"
        artifact.write_text("test")
        
        evaluator.log_to_mlflow(metrics, tmp_path)
        
        assert mock_mlflow.start_run.called
        assert mock_mlflow.log_metric.called

    def test_empty_predictions(self, evaluator):
        """Test handling of empty predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert metrics["mAP_50"] == 0.0 or metrics["precision"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
