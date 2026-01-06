"""Data Processing Pipeline for Heart Disease.

Robust NaN handling + binary target conversion
"""

from __future__ import annotations

import pickle  # noqa: S403
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}")


class HeartDiseasePreprocessor:
    """Preprocessing pipeline."""

    def __init__(self) -> None:  # noqa: D107
        self.scaler: StandardScaler = StandardScaler()
        self.encoders: dict[str, LabelEncoder] = {}
        self.feature_columns: list[str] | None = None
        self.numeric_features: list[str] | None = None
        self.categorical_features: list[str] | None = None
        self.numeric_medians: pd.Series | None = None  # FIXED: PLURAL

    def fit(self, X: pd.DataFrame) -> HeartDiseasePreprocessor:  # noqa: N803 # ML convention
        """Fit on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.

        Returns
        -------
        HeartDiseasePreprocessor
            Fitted preprocessor.

        """
        self.feature_columns = X.columns.tolist()
        self.numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_features = [col for col in X.columns if col not in self.numeric_features]

        # Store medians for NaN filling
        if self.numeric_features:
            self.numeric_medians = X[self.numeric_features].median()
            X_numeric_filled = X[self.numeric_features].fillna(self.numeric_medians)  # noqa: N806 # ML convention
            self.scaler.fit(X_numeric_filled)

        # Fit encoders
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna("missing"))
            self.encoders[col] = le

        logger.info(f"✓ Fitted. Numeric: {len(self.numeric_features)}, Categorical: {len(self.categorical_features)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803 # ML convention
        """Transform with NaN safety.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.

        Returns
        -------
        pd.DataFrame
        Transformed data.

        Raises
        ------
        ValueError
            If output contains NaNs.

        """
        X = X.copy()  # noqa: N806 # ML convention

        # Fix column order
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = np.nan
            X = X.reindex(columns=self.feature_columns)  # noqa: N806 # ML convention

        # Fill numeric NaNs
        if self.numeric_features and self.numeric_medians is not None:
            X[self.numeric_features] = X[self.numeric_features].fillna(self.numeric_medians)

        # Scale numerics
        if self.numeric_features:
            X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])

        # Encode categoricals
        if self.categorical_features is not None:
            for col in self.categorical_features:
                if col in X.columns:
                    X[col] = self.encoders[col].transform(X[col].astype(str).fillna("missing"))

        # FINAL NaN CHECK
        if X.isna().any().any():
            msg = f"Output contains NaNs: {X.isna().sum().sum()}"
            raise ValueError(msg)

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803 # ML convention
        """Fit to data, then transform it.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.

        Returns
        -------
        pd.DataFrame
            Transformed feature data.

        """
        return self.fit(X).transform(X)

    def save(self, filepath: str | Path) -> None:
        """Serialize and save the fitted preprocessor to disk.

        Parameters
        ----------
        filepath : str | pathlib.Path
            Destination path for the serialized preprocessor. Parent directories
            are created if they do not exist.

        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with Path(filepath).open("wb") as f:
            pickle.dump(self, f)
        logger.info("Preprocessor saved: %s", filepath)

    @staticmethod
    def load(filepath: str | Path) -> HeartDiseasePreprocessor:
        """Load a serialized preprocessor from disk.

        Parameters
        ----------
        filepath : str | pathlib.Path
            Path to the serialized preprocessor.

        Returns
        -------
        HeartDiseasePreprocessor
            The deserialized preprocessor instance.

        """
        with Path(filepath).open("rb") as f:
            return pickle.load(f)  # noqa: S301


def load_and_prepare_data(
    raw_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load and prepare Heart Disease dataset.

    Parameters
    ----------
    raw_path : str | pathlib.Path
        Path to the raw CSV data file.
    test_size : float, optional
        Proportion of data to use as test set, by default 0.2
    random_state : int, optional
        Random seed for reproducibility, by default 42

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]
        X_train, X_test, y_train, y_test, feature_columns

    """
    logger.info("Loading: %s", raw_path)

    data = pd.read_csv(raw_path)
    orig_target = data.columns[-1]

    logger.info(f"Original target: {data[orig_target].value_counts().to_dict()}")

    # Binary target: 0=no disease, 1=disease
    data["target_binary"] = (data[orig_target] > 0).astype(int)

    # X = all except BOTH targets
    X = data.drop(columns=[orig_target, "target_binary"])  # noqa: N806 # ML convention
    y = data["target_binary"]

    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806 # ML convention
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()
