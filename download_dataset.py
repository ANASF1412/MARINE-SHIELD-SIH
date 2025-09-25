"""Utility script to download the Roboflow dataset and prepare local structure."""

import logging
import shutil
from pathlib import Path
from typing import Dict

from roboflow import Roboflow

from config import (
    DATASET_DIR,
    DATASET_TRAIN,
    DATASET_VAL,
    DATASET_TEST,
    ROBOFLOW_API_KEY,
    ROBOFLOW_FORMAT,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
    ensure_directories,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def _clean_dataset_dir() -> None:
    if DATASET_DIR.exists():
        logging.info("Existing dataset directory detected. Removing before download...")
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def _verify_dataset_structure() -> Dict[str, Path]:
    required_dirs = {
        "train/images": DATASET_TRAIN,
        "val/images": DATASET_VAL,
        "test/images": DATASET_TEST,
        "train/labels": DATASET_DIR / "train" / "labels",
        "val/labels": DATASET_DIR / "val" / "labels",
        "test/labels": DATASET_DIR / "test" / "labels",
    }

    missing = [name for name, path in required_dirs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset download incomplete. Missing directories: " + ", ".join(missing)
        )
    return required_dirs


def download_dataset() -> None:
    if ROBOFLOW_API_KEY in (None, "", "your_roboflow_api_key_here"):
        raise ValueError(
            "Roboflow API key is not configured. Update ROBOFLOW_API_KEY in config.py or set the environment variable."
        )

    logging.info("Starting dataset download via Roboflow API...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download(ROBOFLOW_FORMAT)

    logging.info("Dataset downloaded. Preparing local directory structure...")
    _clean_dataset_dir()

    downloaded_path = Path(dataset.location)
    for item in downloaded_path.iterdir():
        shutil.move(str(item), DATASET_DIR / item.name)

    _verify_dataset_structure()
    logging.info("Dataset download and verification completed successfully.")


if __name__ == "__main__":
    setup_logging()
    ensure_directories()
    download_dataset()
