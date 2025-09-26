"""PDF report generation utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fpdf import FPDF
from PIL import Image


class DetectionReportBuilder:
	"""Builds PDF reports summarizing segmentation results."""

	def build_report(
		self,
		output_path: Path,
		original_image: Image.Image,
		annotated_array,
		detection_summary: Dict[str, Any],
		predictions: List[Dict[str, Any]] | None = None,
	) -> Path:
		pdf = FPDF(format="A4")
		pdf.set_auto_page_break(auto=True, margin=15)

		# Cover Page
		pdf.add_page()
		pdf.set_font("Arial", "B", 18)
		pdf.cell(0, 12, "Marine Shield - Oil Spill Segmentation Report", ln=True, align="C")
		pdf.set_font("Arial", size=12)
		pdf.cell(0, 10, f"Generated: {datetime.utcnow().isoformat()} UTC", ln=True, align="C")
		pdf.ln(5)

		# Input Image Overview
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 10, "Input Image Overview", ln=True)
		orig_path = output_path.parent / "_temp_original.png"
		original_image.save(orig_path)
		pdf.image(str(orig_path), w=180)

		# Segmentation Results
		pdf.add_page()
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 10, "Segmentation Results", ln=True)
		annotated_path = output_path.parent / "_temp_overlay.png"
		Image.fromarray(annotated_array).save(annotated_path)
		pdf.image(str(annotated_path), w=180)

		# Quantitative Analysis
		pdf.add_page()
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 10, "Quantitative Analysis", ln=True)
		pdf.set_font("Arial", size=12)
		lines = [
			f"Spill detected: {'Yes' if detection_summary.get('spill_detected') else 'No'}",
			f"Total spill area (pixels): {detection_summary.get('total_spill_area', 0):.2f}",
			f"Coverage (% of image): {detection_summary.get('coverage_percent', 0.0):.2f}%",
			f"Shape descriptor: {detection_summary.get('shape', 'N/A')}",
			f"Mean confidence: {detection_summary.get('confidence', 0.0) * 100:.2f}%",
		]
		for ln in lines:
			pdf.cell(0, 8, ln, ln=True)

		# Risk Assessments (simple heuristic)
		pdf.ln(4)
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 10, "Risk Assessment", ln=True)
		pdf.set_font("Arial", size=12)
		coverage = detection_summary.get('coverage_percent', 0.0)
		confidence = detection_summary.get('confidence', 0.0)
		if coverage > 10.0 and confidence > 0.6:
			risk = "High"
		elif coverage > 2.0 and confidence > 0.5:
			risk = "Medium"
		else:
			risk = "Low"
		pdf.cell(0, 8, f"Estimated risk level: {risk}", ln=True)

		# Technical Details
		pdf.add_page()
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 10, "Technical Details", ln=True)
		pdf.set_font("Arial", size=11)
		tech_lines = [
			"Model: DeepLab V3+ (TensorFlow/Keras)",
			"Source: Hugging Face Hub",
			"Preprocessing: resize 512x512, normalization (ImageNet)",
			"Postprocessing: softmax/sigmoid, thresholding, overlay alpha=0.4",
		]
		for tl in tech_lines:
			pdf.cell(0, 7, tl, ln=True)

		# Conclusion and Footer
		pdf.add_page()
		pdf.set_font("Arial", "B", 14)
		pdf.cell(0, 10, "Conclusion", ln=True)
		pdf.set_font("Arial", size=12)
		pdf.multi_cell(0, 7, "This report summarizes segmentation-based detection of possible oil spills using a DeepLab V3+ model. Results should be verified against additional sources if used for operational decisions.")
		pdf.ln(6)
		pdf.set_font("Arial", size=10)
		pdf.cell(0, 6, "Marine Shield - Confidential", ln=True, align="C")

		pdf.output(str(output_path))
		# Cleanup temp images
		orig_path.unlink(missing_ok=True)
		annotated_path.unlink(missing_ok=True)
		return output_path