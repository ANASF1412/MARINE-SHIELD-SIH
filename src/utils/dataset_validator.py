"""Dataset and model validation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import yaml

from config import (
    CLASSES,
    DATASET_DIR,
    DATASET_TEST,
    DATASET_TRAIN,
    DATASET_VAL,
    DATASET_YAML,
    YOLO_MODEL_PATH,
)


def _expect_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at: {path}")
    logging.info("[OK] %s located at %s", description, path)


def validate_model(min_size_bytes: int = 1024) -> bool:
    """Validate that the trained YOLO weights exist and appear non-empty."""
    _expect_path(YOLO_MODEL_PATH, "YOLO model weights")

    file_size = YOLO_MODEL_PATH.stat().st_size
    if file_size < min_size_bytes:
        raise ValueError(
            f"Model weights at {YOLO_MODEL_PATH} appear too small ({file_size} bytes). Did training complete?"
        )

    logging.info("[OK] Model weights verified: %.2f KB", file_size / 1024)
    return True


def validate_yaml(expected_classes: int = len(CLASSES)) -> bool:
    """Validate the dataset YAML file for YOLO training."""
    _expect_path(DATASET_YAML, "Dataset YAML configuration")

    with DATASET_YAML.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    required_keys = {"path", "train", "val", "nc", "names"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Dataset YAML missing required keys: {', '.join(sorted(missing))}")

    if data["nc"] != expected_classes:
        raise ValueError(
            f"Dataset YAML declares {data['nc']} classes, expected {expected_classes}. Update config.py CLASSES."
        )

    yaml_names = [str(name) for name in data.get("names", [])]
    if len(yaml_names) != expected_classes:
        raise ValueError(
            "Dataset YAML 'names' entry does not match expected class count."
        )

    logging.info("[OK] Dataset YAML validated: %s", ", ".join(yaml_names))
    return True


def validate_dataset() -> bool:
    """Validate that dataset directory and split structure exist."""
    _expect_path(DATASET_DIR, "Dataset root directory")

    required_dirs: Dict[str, Path] = {
        "train/images": DATASET_TRAIN,
        "train/labels": DATASET_DIR / "train" / "labels",
        "val/images": DATASET_VAL,
        "val/labels": DATASET_DIR / "val" / "labels",
        "test/images": DATASET_TEST,
        "test/labels": DATASET_DIR / "test" / "labels",
    }

    for description, path in required_dirs.items():
        _expect_path(path, description)
        image_count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png"))) if "images" in description else len(list(path.glob("*.txt")))
        if image_count == 0:
            raise ValueError(f"No files found in {description} ({path}).")
        logging.info("[OK] %s contains %d files", description, image_count)

    logging.info("Dataset validation completed successfully.")
    return True