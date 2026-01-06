"""Data pipeline orchestration."""

import logging
import os

from .loaders import load_combined
from .preprocessing import clean_text, normalize_labels
from .features import engineer_all
from .validation import validate_dataset
from .constants import DATA_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(output_path: str = DATA_PATHS["processed"]) -> None:
    logger.info("Starting data pipeline...")
    df = load_combined()
    df = clean_text(df)
    df = normalize_labels(df)
    df = engineer_all(df)
    validate_dataset(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    os.system(f"dvc add {output_path}")
    logger.info("DVC versioning complete")

if __name__ == "__main__":
    run_pipeline()