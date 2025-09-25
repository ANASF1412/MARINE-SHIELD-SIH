"""PDF report generation utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fpdf import FPDF
from PIL import Image


class DetectionReportBuilder:
    """Builds PDF reports summarizing detection results."""

    def build_report(
        self,
        output_path: Path,
        original_image: Image.Image,
        annotated_array,
        detection_summary: Dict[str, Any],
        predictions: List[Dict[str, Any]],
    ) -> Path:
        pdf = FPDF(format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Oil Spill Detection Report", ln=True, align="C")

        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Generated: {datetime.utcnow().isoformat()} UTC", ln=True)
        pdf.ln(5)

        # Summary section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, self._format_summary(detection_summary))
        pdf.ln(3)

        # Predictions table
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Detections", ln=True)
        pdf.set_font("Arial", size=11)
        if predictions:
            for idx, pred in enumerate(predictions, start=1):
                pdf.multi_cell(
                    0,
                    7,
                    f"#{idx} | Class: {pred['class_name']} | Confidence: {pred['confidence']:.2f} | "
                    f"BBox: {[round(x, 2) for x in pred['bbox']]}"
                )
                pdf.ln(1)
        else:
            pdf.cell(0, 8, "No detections above confidence threshold.", ln=True)

        # Annotated image
        pdf.add_page()
        annotated_path = output_path.parent / "_temp_annotated.png"
        Image.fromarray(annotated_array).save(annotated_path)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Annotated Image", ln=True)
        pdf.image(str(annotated_path), w=180)

        pdf.output(str(output_path))
        annotated_path.unlink(missing_ok=True)
        return output_path

    @staticmethod
    def _format_summary(summary: Dict[str, Any]) -> str:
        lines = [
            f"Spill detected: {'Yes' if summary.get('spill_detected') else 'No'}",
            f"Total spill area (pixels^2): {summary.get('total_spill_area', 0.0):.2f}",
            f"Detection count: {summary.get('detection_count', 0)}",
        ]

        if summary.get("bounding_box"):
            bbox = [round(x, 2) for x in summary["bounding_box"]]
            lines.append(f"Primary bounding box: {bbox}")
            lines.append(f"Confidence: {summary.get('confidence', 0.0):.2f}")

        return "\n".join(lines)