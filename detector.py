"""Vehicle detection module using YOLOv8.

Detection logic is intentionally kept separate from video reading so this
mini-project stays modular and easier to explain in a viva/demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class Detection:
    """One detected vehicle with bounding box and confidence."""

    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        """Bounding box center point used for region assignment."""

        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class YOLOVehicleDetector:
    """Wrapper around Ultralytics YOLO model for traffic classes."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        allowed_classes: tuple[str, ...] = ("car", "bus", "truck", "motorcycle"),
        force_cpu: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = set(allowed_classes)
        self.device: str | None = "cpu" if force_cpu else None

    @staticmethod
    def _resolve_class_name(names: dict[int, str] | list[str], cls_id: int) -> str:
        """Resolve class ID to readable label."""

        if isinstance(names, dict):
            return str(names.get(cls_id, str(cls_id)))
        if cls_id < 0 or cls_id >= len(names):
            return str(cls_id)
        return str(names[cls_id])

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO detection and return only vehicle classes."""

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        detections: list[Detection] = []
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = self._resolve_class_name(names, cls_id)
            if class_name not in self.allowed_classes:
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""

        for detection in detections:
            cv2.rectangle(
                frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                (0, 255, 255),
                2,
            )
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (detection.x1, max(20, detection.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame
