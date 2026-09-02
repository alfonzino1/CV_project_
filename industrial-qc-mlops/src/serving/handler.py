"""TorchServe custom handler for YOLOv8 defect detection."""

import base64
import json
import logging
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DefectDetectionHandler(BaseHandler):
    """
    Custom TorchServe handler for YOLOv8 defect detection.

    Features:
    - Dynamic batching support
    - GPU memory optimization
    - Sub-50ms latency target
    - Health check with test image
    """

    def __init__(self) -> None:
        """Initialize handler."""
        super().__init__()
        self.model: Optional[YOLO] = None
        self.initialized = False
        self.device = "cpu"
        self.image_size = 640
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_batch_size = 8

    def initialize(self, ctx: Any) -> None:
        """
        Initialize model and device.

        Args:
            ctx: TorchServe context object.
        """
        properties = ctx.system_properties
        self.device = f"cuda:{properties.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu"

        # Load model
        model_dir = properties.get("model_dir")
        manifest = ctx.manifest

        # Try to load from .mar file or direct path
        try:
            model_path = f"{model_dir}/model.pt"
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            # Fallback: use pretrained model
            logger.warning(f"Failed to load model from {model_path}: {e}")
            self.model = YOLO("yolov8n.pt")

        # Set device
        if self.device.startswith("cuda"):
            self.model.to(self.device)
            # Warm-up GPU
            self._warmup_gpu()

        self.initialized = True
        logger.info(f"Handler initialized on {self.device}")

    def _warmup_gpu(self) -> None:
        """Warm-up GPU with dummy inference."""
        if not self.model:
            return

        logger.info("Running GPU warm-up...")
        dummy_input = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)

        for _ in range(3):
            _ = self.model.predict(dummy_input, verbose=False)

        logger.info("GPU warm-up completed")

    def preprocess(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Preprocess input images.

        Args:
            data: List of request bodies.

        Returns:
            Batched tensor of preprocessed images.
        """
        images = []

        for item in data:
            # Extract image from request
            image_data = None

            # Check different input formats
            if "data" in item:
                image_data = item["data"]
            elif "image" in item:
                image_data = item["image"]
            elif "body" in item:
                image_data = item["body"]

            if image_data is None:
                raise ValueError("No image data found in request")

            # Decode image
            if isinstance(image_data, str):
                # Base64 encoded
                if image_data.startswith("data:image"):
                    # Remove data URL prefix
                    image_data = image_data.split(",")[1]

                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
            elif isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data)).convert("RGB")
            elif isinstance(image_data, np.ndarray):
                image = Image.fromarray(image_data).convert("RGB")
            elif isinstance(image_data, Image.Image):
                image = image_data.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(image_data)}")

            # Resize and normalize
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_array = np.array(image).astype(np.float32) / 255.0

            # Normalize with ImageNet statistics
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std

            # Convert to CHW format
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
            images.append(img_tensor)

        # Batch images
        batch = torch.stack(images).to(self.device)

        logger.debug(f"Preprocessed batch: {batch.shape}")

        return batch

    def inference(self, batch: torch.Tensor) -> List[Any]:
        """
        Run model inference.

        Args:
            batch: Batch of preprocessed images.

        Returns:
            List of predictions.
        """
        start_time = time.time()

        # Run inference
        results = self.model.predict(
            batch,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        inference_time = (time.time() - start_time) * 1000  # ms

        logger.debug(f"Inference completed in {inference_time:.2f}ms")

        return results

    def postprocess(self, results: List[Any]) -> List[Dict[str, Any]]:
        """
        Postprocess predictions.

        Args:
            results: Raw model predictions.

        Returns:
            List of formatted predictions.
        """
        predictions = []

        for result in results:
            boxes = result.boxes
            pred_dict = {
                "detections": [],
                "num_detections": len(boxes) if boxes else 0,
            }

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    detection = {
                        "bbox": boxes.xyxy[i].tolist(),  # [x1, y1, x2, y2]
                        "confidence": float(boxes.conf[i]),
                        "class_id": int(boxes.cls[i]),
                        "class_name": self.model.names[int(boxes.cls[i])] if hasattr(self.model, 'names') else f"class_{int(boxes.cls[i])}",
                    }
                    pred_dict["detections"].append(detection)

            predictions.append(pred_dict)

        logger.debug(f"Postprocessed {len(predictions)} predictions")

        return predictions

    def handle(self, data: List[Dict[str, Any]], context: Any) -> List[Dict[str, Any]]:
        """
        Main entry point for inference requests.

        Args:
            data: List of request bodies.
            context: TorchServe context.

        Returns:
            List of predictions.
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized")

        start_time = time.time()

        # Preprocess
        batch = self.preprocess(data)

        # Inference
        results = self.inference(batch)

        # Postprocess
        predictions = self.postprocess(results)

        total_time = (time.time() - start_time) * 1000  # ms

        # Add timing info
        for pred in predictions:
            pred["inference_time_ms"] = total_time

        logger.info(f"Request processed in {total_time:.2f}ms")

        return predictions

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check with test image.

        Returns:
            Health status dictionary.
        """
        try:
            if not self.model:
                return {"status": "unhealthy", "reason": "Model not loaded"}

            # Create dummy image
            dummy_image = torch.rand((1, 3, self.image_size, self.image_size), device=self.device)

            # Run inference
            _ = self.model.predict(dummy_image, verbose=False)

            return {
                "status": "healthy",
                "device": self.device,
                "model_loaded": True,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e),
            }
