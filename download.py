"""Data Download Script - Run this ONCE to fetch the Heart Disease dataset."""

import sys

import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}")


def download_heart_disease_data() -> pd.DataFrame:
    """Download Heart Disease UCI dataset.

    Dataset: 303 samples, 14 features, binary classification

    Returns
    -------
    pd.DataFrame
        The downloaded Heart Disease dataset.

    """
    logger.info("Starting Heart Disease dataset download...")

    # Fetch from UCI ML Repository
    heart_disease_df = pd.read_csv("https://archive.ics.uci.edu/static/public/45/data.csv")

    # Save raw data
    raw_path = "data/raw/heart_disease_raw.csv"
    heart_disease_df.to_csv(raw_path, index=False)
    logger.info("✓ Raw data saved: %s", raw_path)

    # Display info
    logger.info(f"Dataset shape: {heart_disease_df.shape}")
    logger.info(f"Features: {list(heart_disease_df.columns[:-1])}")
    logger.info(f"Target: {heart_disease_df.columns[-1]}")
    logger.info(f"\nTarget distribution:\n{heart_disease_df.iloc[:, -1].value_counts()}")
    logger.info(f"Missing values: {heart_disease_df.isna().sum().sum()} total")

    return heart_disease_df


def main() -> int:
    """Run the Heart Disease dataset download workflow.

    Returns
    -------
    int
        Process exit code (0 for success).

    """
    logger.info("-" * 60)
    logger.info("Heart Disease MLOps - Data Download Script")
    logger.info("-" * 60)

    # Step 2: Download data
    _ = download_heart_disease_data()

    logger.info("-" * 60)
    logger.info("✓ Data preparation complete!")
    logger.info("Next: Train models with training script")
    logger.info("-" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
