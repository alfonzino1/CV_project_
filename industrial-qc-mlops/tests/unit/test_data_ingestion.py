"""Unit tests for data ingestion module."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.ingestion import DataIngestion, validate_yolo_format


class TestValidateYOLOFormat:
    """Tests for YOLO format validation."""

    def test_valid_yolo_line(self):
        """Test valid YOLO annotation line."""
        line = "0 0.5 0.5 0.2 0.3"
        assert validate_yolo_format(line) is True

    def test_invalid_class_id(self):
        """Test invalid class ID (negative)."""
        line = "-1 0.5 0.5 0.2 0.3"
        assert validate_yolo_format(line) is False

    def test_invalid_bbox_coordinates(self):
        """Test invalid bounding box coordinates."""
        line = "0 1.5 0.5 0.2 0.3"  # x > 1.0
        assert validate_yolo_format(line) is False

    def test_wrong_number_of_values(self):
        """Test line with wrong number of values."""
        line = "0 0.5 0.5 0.2"  # Missing height
        assert validate_yolo_format(line) is False

    def test_non_numeric_values(self):
        """Test line with non-numeric values."""
        line = "abc 0.5 0.5 0.2 0.3"
        assert validate_yolo_format(line) is False


class TestDataIngestion:
    """Tests for DataIngestion class."""

    @pytest.fixture
    def ingestion(self):
        """Create DataIngestion instance."""
        return DataIngestion()

    @patch('src.data.ingestion.boto3.client')
    def test_download_from_s3(self, mock_boto3, ingestion):
        """Test downloading from S3."""
        mock_client = Mock()
        mock_boto3.return_value = mock_client
        
        ingestion.download_from_s3("s3://bucket/data", "/tmp/data")
        
        assert mock_client.download_file.called

    def test_validate_dataset_structure(self, ingestion, tmp_path):
        """Test dataset structure validation."""
        # Create valid directory structure
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train").mkdir(parents=True)
        
        result = ingestion.validate_dataset_structure(tmp_path)
        
        assert result["valid"] is True
        assert "images" in result["message"]

    def test_validate_dataset_structure_missing_dirs(self, ingestion, tmp_path):
        """Test validation with missing directories."""
        # Create incomplete structure
        (tmp_path / "images").mkdir()
        
        result = ingestion.validate_dataset_structure(tmp_path)
        
        assert result["valid"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
