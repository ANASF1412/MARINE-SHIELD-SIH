"""DeepLab V3+ segmentation model integration using Hugging Face Hub (TensorFlow/Keras)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class SegmentationOutput:
	mask: np.ndarray  # HxW boolean or uint8 mask for oil spill
	prob_map: Optional[np.ndarray]  # HxW float probabilities for oil class
	overlay: Image.Image  # RGB visualization overlayed on original
	area_pixels: int
	confidence: float  # mean prob over mask or 0.0 if none
	shape_descriptor: str


class DeepLabSegmenter:
	"""Wraps a TF/Keras DeepLab V3+ model for semantic segmentation inference."""

	def __init__(
		self,
		hf_repo: str = "Xuanlong/MUAD_DeepLabmodel",
		filename: str = "deeplabv3plus_resnet101_muad_saved_model.zip",
		input_size: Tuple[int, int] = (512, 512),
		class_index_oil: int = 1,
		confidence_threshold: float = 0.5,
	) -> None:
		self.hf_repo = hf_repo
		self.filename = filename
		self.input_size = input_size
		self.class_index_oil = class_index_oil
		self.confidence_threshold = confidence_threshold
		self._model = None

	def _ensure_model(self):
		if self._model is not None:
			return self._model
		# Lazy imports to avoid DLL load errors at process import time
		try:
			import tensorflow as tf  # noqa: F401
			from tensorflow import keras
			from huggingface_hub import hf_hub_download
		except Exception as e:
			LOGGER.error("Required packages not available: %s", e)
			raise
		# Try to enable GPU memory growth if GPU exists
		try:
			gpus = tf.config.list_physical_devices('GPU')
			if gpus:
				for gpu in gpus:
					tf.config.experimental.set_memory_growth(gpu, True)
				LOGGER.info("TensorFlow GPUs available: %s", [d.name for d in gpus])
			else:
				LOGGER.info("No TensorFlow GPU found. Using CPU.")
		except Exception as gpu_e:
			LOGGER.warning("Could not configure TensorFlow GPU memory growth: %s", gpu_e)
		LOGGER.info("Downloading DeepLab V3+ model from Hugging Face: %s/%s", self.hf_repo, self.filename)
		model_path = hf_hub_download(repo_id=self.hf_repo, filename=self.filename, cache_dir=str(Path(".cache/hf").absolute()))
		loaded = keras.models.load_model(model_path)
		self._model = loaded
		return self._model

	def _preprocess(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
		image = image.convert("RGB")
		orig_w, orig_h = image.size
		resized = image.resize(self.input_size[::-1], Image.BILINEAR)
		arr = np.array(resized).astype(np.float32) / 255.0
		arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
		arr = np.expand_dims(arr, axis=0)
		return arr, (orig_h, orig_w)

	def _postprocess(
		self,
		logits: np.ndarray,
		original_hw: Tuple[int, int],
		original_image: Image.Image,
	) -> SegmentationOutput:
		import tensorflow as tf
		# logits shape: (1, H, W, C)
		if logits.ndim == 4 and logits.shape[-1] > 1:
			prob = tf.nn.softmax(logits, axis=-1).numpy()[0]
			prob_oil = prob[..., self.class_index_oil]
		elif logits.ndim == 4:
			prob_oil = tf.math.sigmoid(logits[0, ..., 0]).numpy()
		else:
			raise ValueError("Unexpected logits shape for segmentation output")

		orig_h, orig_w = original_hw
		prob_resized = tf.image.resize(prob_oil[..., None], size=(orig_h, orig_w), method="bilinear").numpy()[..., 0]
		mask = (prob_resized >= self.confidence_threshold).astype(np.uint8)
		area_pixels = int(mask.sum())
		confidence = float(prob_resized[mask == 1].mean()) if area_pixels > 0 else 0.0
		shape_descriptor = "diffuse" if area_pixels > 0 and area_pixels < (0.05 * orig_h * orig_w) else ("extensive" if area_pixels > (0.2 * orig_h * orig_w) else "localized")
		orig_rgb = original_image.convert("RGB")
		overlay = np.array(orig_rgb).astype(np.float32)
		color = np.array([255, 0, 0], dtype=np.float32)
		alpha = 0.4
		overlay[mask == 1] = overlay[mask == 1] * (1 - alpha) + color * alpha
		overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
		return SegmentationOutput(
			mask=mask,
			prob_map=prob_resized,
			overlay=overlay_img,
			area_pixels=area_pixels,
			confidence=confidence,
			shape_descriptor=shape_descriptor,
		)

	def predict(self, image_path: Path) -> SegmentationOutput:
		from tensorflow import keras
		img = Image.open(image_path)
		inp, orig_hw = self._preprocess(img)
		model = self._ensure_model()
		logits = model.predict(inp, verbose=0)
		return self._postprocess(logits, orig_hw, img) 