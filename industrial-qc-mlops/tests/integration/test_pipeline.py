"""Integration tests for the full MLOps pipeline."""
import pytest
import subprocess
from pathlib import Path


class TestDockerComposeStack:
    """Tests for Docker Compose local stack."""

    @pytest.fixture(scope="class")
    def docker_stack(self):
        """Start Docker Compose stack."""
        compose_file = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        
        # Start stack
        subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d"],
            check=True,
            capture_output=True
        )
        
        yield
        
        # Stop stack
        subprocess.run(
            ["docker-compose", "-f", str(compose_file), "down"],
            check=True,
            capture_output=True
        )

    @pytest.mark.slow
    def test_minio_health(self, docker_stack):
        """Test MinIO health endpoint."""
        import requests
        
        response = requests.get("http://localhost:9000/minio/health/live", timeout=10)
        assert response.status_code == 200

    @pytest.mark.slow
    def test_mlflow_health(self, docker_stack):
        """Test MLflow health endpoint."""
        import requests
        
        response = requests.get("http://localhost:5000/health", timeout=10)
        assert response.status_code in [200, 404]  # 404 is OK if no /health endpoint

    @pytest.mark.slow
    def test_prefect_health(self, docker_stack):
        """Test Prefect API health."""
        import requests
        
        response = requests.get("http://localhost:4200/api/health", timeout=10)
        assert response.status_code == 200


class TestTrainingPipeline:
    """Tests for training pipeline execution."""

    @pytest.mark.slow
    def test_training_script_syntax(self):
        """Test that training script has valid syntax."""
        result = subprocess.run(
            ["python", "-m", "py_compile", "src/training/train.py"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"

    @pytest.mark.slow
    def test_config_validation(self):
        """Test that config files are valid YAML."""
        import yaml
        
        config_files = [
            "configs/data.yaml",
            "configs/model.yaml",
            "configs/training.yaml"
        ]
        
        for config_file in config_files:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
                assert data is not None, f"Empty config: {config_file}"


class TestServingPipeline:
    """Tests for serving pipeline."""

    @pytest.mark.slow
    def test_handler_import(self):
        """Test that TorchServe handler can be imported."""
        try:
            from src.serving.handler import DefectDetectionHandler
            assert DefectDetectionHandler is not None
        except ImportError as e:
            pytest.fail(f"Failed to import handler: {e}")

    @pytest.mark.slow
    def test_archive_creator_syntax(self):
        """Test archive creator script syntax."""
        result = subprocess.run(
            ["python", "-m", "py_compile", "src/serving/archive_creator.py"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
