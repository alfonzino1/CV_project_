"""Data preprocessing with Albumentations and YOLO format handling."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pydantic import BaseModel

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger(__name__)


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing pipeline."""
    
    image_size: int = 640
    normalize: bool = True
    mean: List[float] = [0.485, 0.456, 0.406]  # ImageNet mean
    std: List[float] = [0.229, 0.224, 0.225]   # ImageNet std
    max_labels: int = 100
    keep_aspect_ratio: bool = True
    fill_value: int = 114  # Gray value for padding
    
    # Augmentation probabilities
    flip_prob: float = 0.5
    hsv_prob: float = 0.5
    mosaic_prob: float = 0.5
    
    class Config:
        frozen = True


class BoundingBox(BaseModel):
    """YOLO format bounding box."""
    
    class_id: int
    x_center: float  # Normalized 0-1
    y_center: float  # Normalized 0-1
    width: float     # Normalized 0-1
    height: float    # Normalized 0-1
    
    class Config:
        frozen = True


class PreprocessedSample(BaseModel):
    """Result of preprocessing a single sample."""
    
    image_path: str
    image_tensor: Optional[np.ndarray] = None
    boxes: List[BoundingBox] = []
    original_width: int = 0
    original_height: int = 0
    resized_width: int = 0
    resized_height: int = 0


class DataPreprocessor:
    """
    Handle image preprocessing and augmentations for YOLOv8.
    
    Features:
    - Resize with aspect ratio preservation (letterbox)
    - Normalization using ImageNet statistics
    - Albumentations-based augmentations
    - YOLO format annotation parsing and conversion
    - Mosaic augmentation for training
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        settings: Optional[Settings] = None,
        is_training: bool = True,
    ) -> None:
        """
        Initialize DataPreprocessor.
        
        Args:
            config: Preprocessing configuration.
            settings: Application settings.
            is_training: Whether in training mode (enables augmentations).
        """
        self.config = config or PreprocessingConfig()
        self.settings = settings or get_settings()
        self.is_training = is_training
        
        # Build augmentation pipelines
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
        
        logger.info(
            "DataPreprocessor initialized",
            is_training=is_training,
            image_size=self.config.image_size,
        )
    
    def _build_train_transform(self) -> A.Compose:
        """Build training augmentation pipeline."""
        transforms = [
            # Horizontal flip
            A.HorizontalFlip(p=self.config.flip_prob),
            
            # Color augmentations
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=self.config.hsv_prob,
            ),
            
            # Random brightness/contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            
            # Gaussian noise
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            
            # Blur for robustness
            A.MotionBlur(blur_limit=3, p=0.2),
            
            # Resize with letterbox
            A.Resize(
                height=self.config.image_size,
                width=self.config.image_size,
                interpolation=cv2.INTER_LINEAR,
            ),
            
            # Normalize
            A.Normalize(
                mean=self.config.mean,
                std=self.config.std,
                max_pixel_value=255.0,
            ) if self.config.normalize else A.NoOp(),
            
            # Convert to tensor
            ToTensorV2(),
        ]
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.1,
            ),
        )
    
    def _build_val_transform(self) -> A.Compose:
        """Build validation/test augmentation pipeline (minimal)."""
        transforms = [
            # Only resize
            A.Resize(
                height=self.config.image_size,
                width=self.config.image_size,
                interpolation=cv2.INTER_LINEAR,
            ),
            
            # Normalize
            A.Normalize(
                mean=self.config.mean,
                std=self.config.std,
                max_pixel_value=255.0,
            ) if self.config.normalize else A.NoOp(),
            
            # Convert to tensor
            ToTensorV2(),
        ]
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
            ),
        )
    
    def load_image(self, image_path: Path) -> Tuple[np.ndarray, int, int]:
        """
        Load image from disk.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Tuple of (image RGB array, original width, original height).
        """
        try:
            # Use PIL for consistent RGB loading
            img_pil = Image.open(image_path).convert("RGB")
            img_array = np.array(img_pil)
            
            height, width = img_array.shape[:2]
            
            logger.debug(
                "Image loaded",
                path=str(image_path),
                width=width,
                height=height,
            )
            
            return img_array, width, height
            
        except Exception as e:
            logger.error(
                "Failed to load image",
                path=str(image_path),
                error=str(e),
            )
            raise
    
    def parse_yolo_annotation(
        self,
        annotation_path: Path,
        image_width: int,
        image_height: int,
    ) -> List[BoundingBox]:
        """
        Parse YOLO format annotation file.
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All values are normalized to [0, 1].
        
        Args:
            annotation_path: Path to .txt annotation file.
            image_width: Original image width (for validation).
            image_height: Original image height (for validation).
            
        Returns:
            List of BoundingBox objects.
        """
        boxes: List[BoundingBox] = []
        
        if not annotation_path.exists():
            logger.warning("Annotation file not found", path=str(annotation_path))
            return boxes
        
        try:
            with open(annotation_path, "r") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        logger.warning(
                            "Invalid annotation line",
                            file=str(annotation_path),
                            line=line_num,
                        )
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate normalized coordinates
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        logger.warning(
                            "Invalid box coordinates",
                            file=str(annotation_path),
                            line=line_num,
                        )
                        continue
                    
                    if not (0 <= width <= 1 and 0 <= height <= 1):
                        logger.warning(
                            "Invalid box dimensions",
                            file=str(annotation_path),
                            line=line_num,
                        )
                        continue
                    
                    boxes.append(BoundingBox(
                        class_id=class_id,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                    ))
            
            logger.debug(
                "Annotation parsed",
                path=str(annotation_path),
                num_boxes=len(boxes),
            )
            
        except Exception as e:
            logger.error(
                "Failed to parse annotation",
                path=str(annotation_path),
                error=str(e),
            )
        
        return boxes[:self.config.max_labels]
    
    def apply_mosaic(
        self,
        images: List[np.ndarray],
        boxes_list: List[List[BoundingBox]],
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """
        Apply mosaic augmentation (4 images combined).
        
        Args:
            images: List of 4 images.
            boxes_list: List of bounding boxes for each image.
            
        Returns:
            Tuple of (mosaic image, combined boxes in new coordinates).
        """
        if len(images) != 4:
            raise ValueError("Mosaic requires exactly 4 images")
        
        # Get output size
        s = self.config.image_size
        
        # Place 4 images in corners
        yc, xc = [int(random.uniform(s // 2, s)) for _ in range(2)]
        
        mosaic = np.full((s * 2, s * 2, 3), self.config.fill_value, dtype=np.uint8)
        
        new_boxes = []
        
        for i, (img, boxes) in enumerate(zip(images, boxes_list)):
            h, w = img.shape[:2]
            
            # Resize image to fit quadrant
            scale = min(s / h, s / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Calculate position in mosaic
            if i == 0:  # Top-left
                x1a, y1a = max(xc - new_w, 0), max(yc - new_h, 0)
                x1b, y1b = new_w - (xc - x1a), new_h - (yc - y1a)
                x2a, y2a = xc, yc
                x2b, y2b = xc + (new_w - x1b), yc + (new_h - y1b)
            elif i == 1:  # Top-right
                x1a, y1a = xc, max(yc - new_h, 0)
                x1b, y1b = 0, new_h - (yc - y1a)
                x2a, y2a = min(xc + new_w, s * 2), yc
                x2b, y2b = x1b + (x2a - x1a), yc + (new_h - y1b)
            elif i == 2:  # Bottom-left
                x1a, y1a = max(xc - new_w, 0), yc
                x1b, y1b = new_w - (xc - x1a), 0
                x2a, y2a = xc, min(yc + new_h, s * 2)
                x2b, y2b = xc + (new_w - x1b), y1b + (y2a - y1a)
            else:  # Bottom-right
                x1a, y1a = xc, yc
                x1b, y1b = 0, 0
                x2a, y2a = min(xc + new_w, s * 2), min(yc + new_h, s * 2)
                x2b, y2b = x1b + (x2a - x1a), y1b + (y2a - y1a)
            
            # Place image in mosaic
            if y2a > y1a and x2a > x1a:
                mosaic[y1a:y2a, x1a:x2a] = img_resized[y1b:y2b, x1b:x2b]
                
                # Transform boxes
                for box in boxes:
                    # Denormalize to original image
                    x1 = (box.x_center - box.width / 2) * w
                    y1 = (box.y_center - box.height / 2) * h
                    x2 = (box.x_center + box.width / 2) * w
                    y2 = (box.y_center + box.height / 2) * h
                    
                    # Scale to resized image
                    x1 *= scale
                    y1 *= scale
                    x2 *= scale
                    y2 *= scale
                    
                    # Shift to mosaic position
                    shift_x = x1a - x1b if i == 0 else (xc if i == 1 else (x1a if i == 2 else xc))
                    shift_y = y1a - y1b if i == 0 else (yc if i == 0 else (yc if i == 1 else y1a))
                    
                    x1 += shift_x
                    y1 += shift_y
                    x2 += shift_x
                    y2 += shift_y
                    
                    # Check if box is still valid
                    if x2 > x1 and y2 > y1:
                        # Normalize to mosaic
                        new_x_center = (x1 + x2) / 2 / (s * 2)
                        new_y_center = (y1 + y2) / 2 / (s * 2)
                        new_width = (x2 - x1) / (s * 2)
                        new_height = (y2 - y1) / (s * 2)
                        
                        if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1:
                            if 0 <= new_width <= 1 and 0 <= new_height <= 1:
                                new_boxes.append(BoundingBox(
                                    class_id=box.class_id,
                                    x_center=new_x_center,
                                    y_center=new_y_center,
                                    width=new_width,
                                    height=new_height,
                                ))
        
        return mosaic, new_boxes
    
    def preprocess_sample(
        self,
        image_path: Path,
        annotation_path: Optional[Path] = None,
    ) -> PreprocessedSample:
        """
        Preprocess a single image sample.
        
        Args:
            image_path: Path to image file.
            annotation_path: Path to YOLO annotation file.
            
        Returns:
            PreprocessedSample with image tensor and boxes.
        """
        # Load image
        img, orig_w, orig_h = self.load_image(image_path)
        
        # Parse annotations
        boxes: List[BoundingBox] = []
        if annotation_path and annotation_path.exists():
            boxes = self.parse_yolo_annotation(annotation_path, orig_w, orig_h)
        
        # Prepare boxes for albumentations
        bboxes = [[b.x_center, b.y_center, b.width, b.height] for b in boxes]
        class_labels = [b.class_id for b in boxes]
        
        # Apply transformations
        transform = self.train_transform if self.is_training else self.val_transform
        
        try:
            transformed = transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            
            img_tensor = transformed["image"]
            new_bboxes = transformed["bboxes"]
            new_labels = transformed["class_labels"]
            
            # Convert back to BoundingBox format
            processed_boxes = [
                BoundingBox(
                    class_id=int(label),
                    x_center=b[0],
                    y_center=b[1],
                    width=b[2],
                    height=b[3],
                )
                for b, label in zip(new_bboxes, new_labels)
            ]
            
        except Exception as e:
            logger.warning(
                "Transform failed, using original image",
                path=str(image_path),
                error=str(e),
            )
            # Fallback: just resize without augmentations
            img_resized = cv2.resize(img, (self.config.image_size, self.config.image_size))
            if self.config.normalize:
                img_resized = (img_resized.astype(np.float32) / 255.0 - np.array(self.config.mean)) / np.array(self.config.std)
            img_tensor = img_resized.transpose(2, 0, 1)
            processed_boxes = boxes
        
        return PreprocessedSample(
            image_path=str(image_path),
            image_tensor=img_tensor,
            boxes=processed_boxes,
            original_width=orig_w,
            original_height=orig_h,
            resized_width=self.config.image_size,
            resized_height=self.config.image_size,
        )
    
    def create_dataset_yaml(
        self,
        output_path: Path,
        train_dir: Path,
        val_dir: Path,
        test_dir: Optional[Path] = None,
    ) -> Path:
        """
        Create YOLO dataset YAML configuration.
        
        Args:
            output_path: Output path for YAML file.
            train_dir: Path to training images directory.
            val_dir: Path to validation images directory.
            test_dir: Path to test images directory (optional).
            
        Returns:
            Path to created YAML file.
        """
        import yaml
        
        classes = self.settings.defect_classes
        
        data_config = {
            "path": str(train_dir.parent.absolute()),
            "train": str(train_dir.relative_to(train_dir.parent)),
            "val": str(val_dir.relative_to(val_dir.parent)),
            "test": str(test_dir.relative_to(test_dir.parent)) if test_dir else None,
            "nc": len(classes),
            "names": classes,
        }
        
        # Remove None values
        data_config = {k: v for k, v in data_config.items() if v is not None}
        
        with open(output_path, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(
            "Dataset YAML created",
            path=str(output_path),
            num_classes=len(classes),
        )
        
        return output_path
