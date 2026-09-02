"""Model export and TorchServe archive creation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from ultralytics import YOLO

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger(__name__)


class ModelExporter:
    """
    Export trained YOLOv8 models for production serving.

    Supported formats:
    - TorchScript (.pt)
    - ONNX (.onnx)
    - TensorRT (.engine)
    - TorchServe (.mar)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize exporter.

        Args:
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        self.models_dir = Path(self.settings.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ModelExporter initialized", models_dir=str(self.models_dir))

    def export_to_torchscript(
        self,
        model_path: str,
        output_path: Optional[Path] = None,
        optimize: bool = True,
    ) -> Path:
        """
        Export model to TorchScript format.

        Args:
            model_path: Path to trained YOLOv8 weights.
            output_path: Output path for .pt file.
            optimize: Apply optimizations.

        Returns:
            Path to exported model.
        """
        logger.info("Exporting to TorchScript", model_path=model_path)

        # Load model
        model = YOLO(model_path)

        # Set output path
        if output_path is None:
            output_path = self.models_dir / "model.torchscript.pt"
        else:
            output_path = Path(output_path)

        # Export
        try:
            exported_path = model.export(
                format="torchscript",
                imgsz=self.settings.image_size,
                device=self.settings.gpu_ids.split(",")[0] if self.settings.gpu_ids != "-1" else "cpu",
                optimize=optimize,
            )

            logger.info("TorchScript export completed", path=exported_path)

            return Path(exported_path)

        except Exception as e:
            logger.error("TorchScript export failed", error=str(e), exc_info=True)
            raise

    def export_to_onnx(
        self,
        model_path: str,
        output_path: Optional[Path] = None,
        opset_version: int = 12,
        dynamic_axes: bool = True,
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            model_path: Path to trained YOLOv8 weights.
            output_path: Output path for .onnx file.
            opset_version: ONNX opset version.
            dynamic_axes: Enable dynamic batch size.

        Returns:
            Path to exported model.
        """
        logger.info("Exporting to ONNX", model_path=model_path)

        # Load model
        model = YOLO(model_path)

        # Set output path
        if output_path is None:
            output_path = self.models_dir / "model.onnx"
        else:
            output_path = Path(output_path)

        # Export
        try:
            exported_path = model.export(
                format="onnx",
                imgsz=self.settings.image_size,
                device=self.settings.gpu_ids.split(",")[0] if self.settings.gpu_ids != "-1" else "cpu",
                opset=opset_version,
                dynamic=dynamic_axes,
            )

            logger.info("ONNX export completed", path=exported_path)

            return Path(exported_path)

        except Exception as e:
            logger.error("ONNX export failed", error=str(e), exc_info=True)
            raise

    def quantize_model(
        self,
        model_path: str,
        output_path: Optional[Path] = None,
        mode: str = "fp16",
    ) -> Path:
        """
        Quantize model for faster inference.

        Args:
            model_path: Path to model weights.
            output_path: Output path for quantized model.
            mode: Quantization mode (fp16, int8).

        Returns:
            Path to quantized model.
        """
        mode = mode.lower()
        if mode not in ["fp16", "int8", "none"]:
            raise ValueError(f"Invalid quantization mode: {mode}")

        if mode == "none":
            logger.info("Skipping quantization")
            return Path(model_path)

        logger.info(f"Quantizing model to {mode}", model_path=model_path)

        # Load model
        model = YOLO(model_path)

        # Set output path
        if output_path is None:
            output_path = self.models_dir / f"model_{mode}.pt"
        else:
            output_path = Path(output_path)

        try:
            if mode == "fp16":
                # FP16 quantization (half precision)
                model.model.half()
                torch.save(model.model, str(output_path))
                logger.info("FP16 quantization completed", path=str(output_path))

            elif mode == "int8":
                # INT8 quantization requires calibration
                # Using ONNX Runtime for INT8
                onnx_path = self.export_to_onnx(model_path)

                try:
                    import onnxruntime as ort
                    from onnxruntime.quantization import quantize_dynamic, QuantType

                    quantized_path = output_path.with_suffix(".quantized.onnx")
                    quantize_dynamic(
                        onnx_path,
                        str(quantized_path),
                        weight_type=QuantType.QUInt8,
                    )
                    logger.info("INT8 quantization completed", path=str(quantized_path))
                    output_path = quantized_path

                except ImportError:
                    logger.warning("ONNX Runtime not available, skipping INT8 quantization")
                    output_path = Path(model_path)

            return output_path

        except Exception as e:
            logger.error("Quantization failed", error=str(e), exc_info=True)
            return Path(model_path)

    def create_torchserve_archive(
        self,
        model_path: str,
        handler_path: Path,
        output_path: Optional[Path] = None,
        model_name: str = "defect_detector",
        version: str = "1.0.0",
    ) -> Path:
        """
        Create TorchServe model archive (.mar file).

        Args:
            model_path: Path to model weights.
            handler_path: Path to custom handler.
            output_path: Output path for .mar file.
            model_name: Model name for serving.
            version: Model version.

        Returns:
            Path to .mar file.
        """
        logger.info(
            "Creating TorchServe archive",
            model_path=model_path,
            handler_path=str(handler_path),
        )

        # Set output path
        if output_path is None:
            output_path = self.models_dir / f"{model_name}.mar"
        else:
            output_path = Path(output_path)

        try:
            from torch_model_archiver import cmd_args_parser
            from ts.model_archiver import model_packaging

            # Prepare arguments
            args = cmd_args_parser.get_args([
                "--model-name", model_name,
                "--handler", str(handler_path),
                "--serialized-file", model_path,
                "--model-file", "",  # No separate model file for YOLO
                "--version", version,
                "--export-path", str(output_path.parent),
                "--force",
            ])

            # Create archive
            model_packaging.package_model(args)

            logger.info("TorchServe archive created", path=str(output_path))

            return output_path

        except Exception as e:
            logger.error("Archive creation failed", error=str(e), exc_info=True)
            raise

    def benchmark_inference(
        self,
        model_path: str,
        num_runs: int = 100,
        batch_sizes: List[int] = [1, 4, 8, 16],
    ) -> Dict[str, Any]:
        """
        Benchmark inference latency.

        Args:
            model_path: Path to model.
            num_runs: Number of runs per batch size.
            batch_sizes: Batch sizes to test.

        Returns:
            Dictionary with benchmark results.
        """
        import time

        logger.info("Running inference benchmark", model_path=model_path)

        # Load model
        model = YOLO(model_path)

        # Create dummy input
        device = self.settings.gpu_ids.split(",")[0] if self.settings.gpu_ids != "-1" else "cpu"
        img_size = self.settings.image_size

        results = {}

        for batch_size in batch_sizes:
            latencies = []

            for _ in range(num_runs):
                # Create random image batch
                dummy_input = torch.rand(batch_size, 3, img_size, img_size, device=device)

                # Warm-up first run
                if _ == 0:
                    _ = model.predict(dummy_input, verbose=False)
                    continue

                # Measure latency
                start = time.time()
                _ = model.predict(dummy_input, verbose=False)
                end = time.time()

                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)

            # Calculate statistics
            latencies_array = torch.tensor(latencies)
            results[batch_size] = {
                "mean_ms": float(latencies_array.mean()),
                "median_ms": float(latencies_array.median()),
                "p95_ms": float(torch.quantile(latencies_array, 0.95)),
                "p99_ms": float(torch.quantile(latencies_array, 0.99)),
                "std_ms": float(latencies_array.std()),
                "throughput_fps": float(batch_size / (latencies_array.mean() / 1000)),
            }

            logger.info(
                f"Batch size {batch_size}: "
                f"mean={results[batch_size]['mean_ms']:.2f}ms, "
                f"p95={results[batch_size]['p95_ms']:.2f}ms, "
                f"throughput={results[batch_size]['throughput_fps']:.1f} FPS"
            )

        return results


def export_model(
    model_path: str,
    output_dir: Optional[Path] = None,
    quantization_mode: str = "fp16",
    create_mar: bool = True,
) -> Dict[str, Path]:
    """
    Convenience function to export model in all formats.

    Args:
        model_path: Path to trained model.
        output_dir: Output directory.
        quantization_mode: Quantization mode.
        create_mar: Whether to create TorchServe archive.

    Returns:
        Dictionary with paths to exported models.
    """
    settings = get_settings()
    exporter = ModelExporter(settings=settings)

    output_dir = output_dir or Path(settings.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_models = {}

    # Export to TorchScript
    try:
        ts_path = exporter.export_to_torchscript(
            model_path,
            output_dir / "model.torchscript.pt",
        )
        exported_models["torchscript"] = ts_path
    except Exception as e:
        logger.warning("TorchScript export failed", error=str(e))

    # Export to ONNX
    try:
        onnx_path = exporter.export_to_onnx(
            model_path,
            output_dir / "model.onnx",
        )
        exported_models["onnx"] = onnx_path
    except Exception as e:
        logger.warning("ONNX export failed", error=str(e))

    # Quantize
    try:
        quantized_path = exporter.quantize_model(
            model_path,
            output_dir / f"model_{quantization_mode}.pt",
            mode=quantization_mode,
        )
        exported_models["quantized"] = quantized_path
    except Exception as e:
        logger.warning("Quantization failed", error=str(e))

    # Create TorchServe archive
    if create_mar:
        try:
            handler_path = Path("src/serving/handler.py")
            mar_path = exporter.create_torchserve_archive(
                str(quantized_path) if "quantized" in exported_models else model_path,
                handler_path,
                output_dir / "defect_detector.mar",
            )
            exported_models["mar"] = mar_path
        except Exception as e:
            logger.warning("MAR creation failed", error=str(e))

    # Run benchmark
    try:
        benchmark_results = exporter.benchmark_inference(
            str(exported_models.get("quantized", model_path)),
        )
        exported_models["benchmark"] = benchmark_results
    except Exception as e:
        logger.warning("Benchmark failed", error=str(e))

    return exported_models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLOv8 model")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quantization", type=str, default="fp16", choices=["fp16", "int8", "none"])
    parser.add_argument("--no-mar", action="store_true", help="Skip MAR creation")

    args = parser.parse_args()

    results = export_model(
        model_path=args.model,
        output_dir=Path(args.output) if args.output else None,
        quantization_mode=args.quantization,
        create_mar=not args.no_mar,
    )

    print("\n=== Export Results ===")
    for format_name, path in results.items():
        if format_name != "benchmark":
            print(f"{format_name}: {path}")

    if "benchmark" in results:
        print("\n=== Benchmark Results ===")
        for batch_size, metrics in results["benchmark"].items():
            print(f"\nBatch size: {batch_size}")
            print(f"  Mean latency: {metrics['mean_ms']:.2f} ms")
            print(f"  P95 latency: {metrics['p95_ms']:.2f} ms")
            print(f"  Throughput: {metrics['throughput_fps']:.1f} FPS")
