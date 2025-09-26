import io
import logging
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is on sys.path for top-level config imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DETECTIONS_FOLDER, REPORTS_FOLDER, ensure_directories
from src.models.segmentation import DeepLabSegmenter
from src.utils.reports import DetectionReportBuilder
import tempfile
import time

# Configure
st.set_page_config(page_title="Marine Shield - Oil Spill Segmentation", layout="wide")
logging.basicConfig(level=logging.INFO)

# Ensure directories
ensure_directories()
DETECTIONS_FOLDER.mkdir(parents=True, exist_ok=True)
REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Sidebar controls
st.sidebar.header("Settings")
resize_opt = st.sidebar.selectbox("Inference image size", options=[256, 384, 512, 640, 768], index=2)

# State
@st.cache_resource(show_spinner=False)
def get_segmenter(input_size: int) -> DeepLabSegmenter:
	return DeepLabSegmenter(input_size=(input_size, input_size))

st.session_state.segmenter = get_segmenter(resize_opt)
if "last_overlay" not in st.session_state:
	st.session_state.last_overlay = None
if "last_summary" not in st.session_state:
	st.session_state.last_summary = None
if "last_image_name" not in st.session_state:
	st.session_state.last_image_name = None

st.title("Marine Shield - Oil Spill Segmentation")
st.write("Upload a satellite/ocean image to detect potential oil spills using DeepLab V3+.")

uploaded = st.file_uploader("Select an image", type=["png", "jpg", "jpeg", "tif", "tiff"])

col1, col2 = st.columns(2)

def _show_tf_troubleshooting(error_message: str) -> None:
	st.error("TensorFlow failed to load. See troubleshooting below.")
	with st.expander("Troubleshooting steps (TensorFlow DLL load failure)"):
		st.markdown("- **Verify Python version**: TensorFlow 2.20 supports Python 3.9–3.12.")
		st.markdown("- **Install Microsoft Visual C++ Redistributable (x64)**: `vs_BuildTools.exe` or `VC_redist.x64.exe`.")
		st.markdown("- **If using GPU**: match CUDA/cuDNN versions to TensorFlow; else use CPU TensorFlow.")
		st.markdown("- **Reinstall TensorFlow**: `pip uninstall -y tensorflow` then `pip install tensorflow`. Restart the terminal.")
		st.markdown("- **Check PATH**: ensure no conflicting DLLs; prefer a clean venv.")
		st.markdown("- **Official guide**: see `https://www.tensorflow.org/install/errors`.")
		st.code(error_message)

if uploaded is not None:
	image_name = uploaded.name
	img = Image.open(uploaded).convert("RGB")
	st.session_state.last_image_name = image_name

	with st.status("Running segmentation...", expanded=False) as status:
		status.update(label="Preprocessing image...")
		t0 = time.time()
		segmenter: DeepLabSegmenter = st.session_state.segmenter
		status.update(label="Downloading/loading model (first run may take a while)...")
		# Save upload to a temporary file on disk for inference
		suffix = Path(image_name).suffix or ".png"
		with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
			tmp.write(uploaded.getbuffer())
			tmp_path = Path(tmp.name)
		try:
			seg = segmenter.predict(image_path=tmp_path)
		except ImportError as e:
			# TensorFlow import/runtime error handling
			_show_tf_troubleshooting(str(e))
			# Ensure temp cleanup
			try:
				tmp_path.unlink(missing_ok=True)
			except Exception:
				pass
			st.stop()
		except Exception as e:
			st.error(f"Segmentation failed: {e}")
			try:
				tmp_path.unlink(missing_ok=True)
			except Exception:
				pass
			st.stop()
		finally:
			# Cleanup temp file
			try:
				tmp_path.unlink(missing_ok=True)
			except Exception:
				pass
		t1 = time.time()
		status.update(label="Postprocessing results...")

		# Save overlay for download and later report
		overlay_path = DETECTIONS_FOLDER / f"overlay_{image_name}"
		seg.overlay.save(overlay_path)
		st.session_state.last_overlay = overlay_path

		# Compute summary metrics
		arr = np.array(img)
		h, w = arr.shape[:2]
		coverage_pct = (seg.area_pixels / float(h * w)) * 100.0
		summary = {
			"spill_detected": seg.area_pixels > 0,
			"total_spill_area": float(seg.area_pixels),
			"coverage_percent": coverage_pct,
			"shape": seg.shape_descriptor,
			"confidence": float(seg.confidence),
			"inference_ms": int((t1 - t0) * 1000),
		}
		st.session_state.last_summary = summary
		status.update(label="Done.")
		status.update(state="complete")

	# Display
	with col1:
		st.subheader("Original")
		st.image(img, use_column_width=True)
	with col2:
		st.subheader("Segmentation Overlay")
		st.image(str(st.session_state.last_overlay), use_column_width=True)

	st.subheader("Metrics")
	if st.session_state.last_summary["spill_detected"]:
		st.metric("Area (pixels)", f"{st.session_state.last_summary['total_spill_area']:.0f}")
		st.metric("Coverage (%)", f"{st.session_state.last_summary['coverage_percent']:.2f}")
		st.metric("Confidence (%)", f"{st.session_state.last_summary['confidence']*100:.2f}")
		st.metric("Inference (ms)", f"{st.session_state.last_summary['inference_ms']}")
		st.write(f"Shape: {st.session_state.last_summary['shape']}")
	else:
		st.info("No oil spill detected above threshold.")

	# Report generation and download
	st.subheader("Report")
	if st.button("Generate PDF Report"):
		with st.spinner("Generating report..."):
			t2 = time.time()
			builder = DetectionReportBuilder()
			pdf_path = REPORTS_FOLDER / f"{Path(image_name).stem}_report.pdf"
			builder.build_report(
				output_path=pdf_path,
				original_image=img,
				annotated_array=np.array(Image.open(st.session_state.last_overlay).convert("RGB")),
				detection_summary=st.session_state.last_summary,
			)
			t3 = time.time()
		st.success(f"Report ready in {int((t3 - t2) * 1000)} ms")
		with open(pdf_path, "rb") as f:
			st.download_button("Download Report", data=f.read(), file_name=pdf_path.name, mime="application/pdf")

st.caption("Model: DeepLab V3+ via Hugging Face | Report powered by FPDF | Marine Shield") 