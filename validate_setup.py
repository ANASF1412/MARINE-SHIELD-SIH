"""Validate dataset availability, YAML configuration, and model weights."""

import logging
import sys
from pathlib import Path

from config import LOG_FILE, LOG_LEVEL, ensure_directories
from src.utils.dataset_validator import validate_dataset, validate_model, validate_yaml


def configure_logging() -> None:
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a"),
        ],
    )


def main() -> int:
    configure_logging()
    ensure_directories()

    try:
        logging.info("Validating dataset YAML configuration...")
        validate_yaml()

        logging.info("Validating dataset directory structure...")
        validate_dataset()

        logging.info("Validating model weights...")
        validate_model()

        logging.info("All validations completed successfully!")
        return 0
    except Exception as exc:  # pragma: no cover - logging handles details
        logging.exception("Validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
