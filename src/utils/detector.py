"""High-level detection utilities for Flask routes and PDF reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from config import DETECTIONS_FOLDER, REPORTS_FOLDER
from src.models.segmentation import DeepLabSegmenter, SegmentationOutput
from src.utils.reports import DetectionReportBuilder

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionResult:
	image_name: str
	annotated_image_path: Path
	summary: Dict[str, Any]
	pdf_report_path: Path | None = None


class DetectionManager:
	"""Handles detection workflow including image processing and report generation."""

	def __init__(self) -> None:
		self.segmenter = DeepLabSegmenter()
		self.report_builder = DetectionReportBuilder()

	def process_image(self, image_path: Path) -> DetectionResult:
		LOGGER.info("Processing image (segmentation): %s", image_path)
		image_path = Path(image_path)
		DETECTIONS_FOLDER.mkdir(parents=True, exist_ok=True)

		seg: SegmentationOutput = self.segmenter.predict(image_path)
		annotated_output = DETECTIONS_FOLDER / f"overlay_{image_path.name}"
		seg.overlay.save(annotated_output)

		summary = self._summarize_segmentation(seg, image_path)
		return DetectionResult(
			image_name=image_path.name,
			annotated_image_path=annotated_output,
			summary=summary,
		)

	def build_pdf_report(self, detection_result: DetectionResult) -> Path:
		LOGGER.info("Generating PDF report for %s", detection_result.image_name)
		pdf_path = REPORTS_FOLDER / f"{Path(detection_result.image_name).stem}_report.pdf"
		REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

		annotated_image_array = np.array(Image.open(detection_result.annotated_image_path).convert("RGB"))
		self.report_builder.build_report(
			output_path=pdf_path,
			original_image=Image.open(detection_result.annotated_image_path),
			annotated_array=annotated_image_array,
			detection_summary=detection_result.summary,
			predictions=[],
		)

		detection_result.pdf_report_path = pdf_path
		return pdf_path

	@staticmethod
	def _summarize_segmentation(seg: SegmentationOutput, image_path: Path) -> Dict[str, Any]:
		# Pixel area and simple metrics
		area_pixels = seg.area_pixels
		spill_detected = area_pixels > 0
		confidence = seg.confidence

		# Estimate area in sq km if GSD unknown: leave pixels^2 and percentage
		overlay_img = Image.open(image_path).convert("RGB")
		h, w = np.array(overlay_img).shape[:2]
		coverage_pct = (area_pixels / float(h * w)) * 100.0

		return {
			"spill_detected": spill_detected,
			"total_spill_area": float(area_pixels),
			"coverage_percent": coverage_pct,
			"shape": seg.shape_descriptor,
			"confidence": float(confidence),
		}