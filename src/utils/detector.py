"""High-level detection utilities for Flask routes and PDF reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

from config import DETECTIONS_FOLDER, REPORTS_FOLDER
from src.models.yolo_model import YOLOModelManager
from src.utils.reports import DetectionReportBuilder

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    image_name: str
    annotated_image_path: Path
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]
    pdf_report_path: Path | None = None


class DetectionManager:
    """Handles detection workflow including image processing and report generation."""

    def __init__(self) -> None:
        self.model_manager = YOLOModelManager()
        self.report_builder = DetectionReportBuilder()

    def process_image(self, image_path: Path) -> DetectionResult:
        LOGGER.info("Processing image: %s", image_path)

        predictions = self.model_manager.predict(image_path)
        LOGGER.debug("Predictions: %s", predictions)

        output_image = DETECTIONS_FOLDER / f"annotated_{image_path.name}"
        annotated_path = self.model_manager.render_predictions(image_path, output_image)

        summary = self._summarize_predictions(predictions)
        return DetectionResult(
            image_name=image_path.name,
            annotated_image_path=annotated_path,
            predictions=predictions,
            summary=summary,
        )

    def build_pdf_report(self, detection_result: DetectionResult) -> Path:
        LOGGER.info("Generating PDF report for %s", detection_result.image_name)
        pdf_path = REPORTS_FOLDER / f"{Path(detection_result.image_name).stem}_report.pdf"
        REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

        annotated_image_array = self.model_manager.load_image(detection_result.annotated_image_path)
        self.report_builder.build_report(
            output_path=pdf_path,
            original_image=Image.open(detection_result.annotated_image_path),
            annotated_array=annotated_image_array,
            detection_summary=detection_result.summary,
            predictions=detection_result.predictions,
        )

        detection_result.pdf_report_path = pdf_path
        return pdf_path

    @staticmethod
    def _summarize_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_spill_area = 0.0
        spill_detections = []

        for pred in predictions:
            bbox = pred["bbox"]
            width = max(0.0, bbox[2] - bbox[0])
            height = max(0.0, bbox[3] - bbox[1])
            area = width * height
            pred["area"] = area

            if pred["class_name"] == "oil_spill":
                total_spill_area += area
                spill_detections.append(pred)

        primary_detection = spill_detections[0] if spill_detections else None

        summary = {
            "spill_detected": bool(spill_detections),
            "total_spill_area": total_spill_area,
            "detection_count": len(spill_detections),
            "bounding_box": primary_detection["bbox"] if primary_detection else None,
            "confidence": primary_detection["confidence"] if primary_detection else 0.0,
            "class_name": primary_detection["class_name"] if primary_detection else None,
        }

        LOGGER.debug("Detection summary: %s", summary)
        return summary