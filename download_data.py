#!/usr/bin/env python3
"""Data Download Script - Run this ONCE to fetch the Heart Disease dataset."""

import logging
import sys
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """Create necessary directory structure"""
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "mlruns",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info("✓ Directory ensured: %s", dir_path)


def download_heart_disease_data():
    """Download Heart Disease UCI dataset
    Dataset: 303 samples, 14 features, binary classification
    """
    logger.info("Starting Heart Disease dataset download...")

    try:
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
        logger.info(f"Missing values: {heart_disease_df.isnull().sum().sum()} total")

        return heart_disease_df

    except Exception as e:
        logger.error(f"✗ Download failed: {e!s}")
        logger.info("Fallback: Creating sample dataset...")
        return create_sample_dataset()


def create_sample_dataset():
    """Create sample dataset if download fails"""
    logger.warning("Creating sample dataset for development...")

    import numpy as np

    np.random.seed(42)

    n_samples = 303
    features = {
        "age": np.random.randint(29, 78, n_samples),
        "sex": np.random.randint(0, 2, n_samples),
        "cp": np.random.randint(0, 4, n_samples),
        "trestbps": np.random.randint(94, 200, n_samples),
        "chol": np.random.randint(126, 565, n_samples),
        "fbs": np.random.randint(0, 2, n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalach": np.random.randint(71, 202, n_samples),
        "exang": np.random.randint(0, 2, n_samples),
        "oldpeak": np.random.uniform(0, 6.2, n_samples),
        "slope": np.random.randint(0, 3, n_samples),
        "ca": np.random.randint(0, 5, n_samples),
        "thal": np.random.randint(0, 4, n_samples),
        "num": np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(features)
    raw_path = "data/raw/heart_disease_raw.csv"
    df.to_csv(raw_path, index=False)

    logger.info("✓ Sample data created: %s", raw_path)
    return df


def main():
    logger.info("=" * 60)
    logger.info("Heart Disease MLOps - Data Download Script")
    logger.info("=" * 60)

    # Step 1: Ensure directories
    ensure_directories()

    # Step 2: Download data
    df = download_heart_disease_data()

    logger.info("=" * 60)
    logger.info("✓ Data preparation complete!")
    logger.info("Next: Train models with training script")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
