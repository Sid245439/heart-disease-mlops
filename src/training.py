"""Model Training & Evaluation.

This module defines the ModelTrainer class, which handles training and evaluating
multiple machine learning models on the Heart Disease UCI dataset. It includes methods
for training Logistic Regression and Random Forest models with hyperparameter tuning,
evaluating their performance using various metrics, and saving the best-performing model.
"""

from __future__ import annotations

import pickle  # noqa: S403
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from src.preprocessing import HeartDiseasePreprocessor, load_and_prepare_data

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}")


class ModelTrainer:
    """Train and evaluate multiple models."""

    def __init__(self, experiment_name: str = "heart-disease-mlops") -> None:
        self.experiment_name = experiment_name
        self.models: dict[str, dict[str, Any]] = {}
        self.best_model: BaseEstimator | None = None
        mlflow.set_experiment(experiment_name)

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,  # noqa: N803
        y_train: pd.Series,
        X_test: pd.DataFrame,  # noqa: N803
        y_test: pd.Series,
    ) -> tuple[LogisticRegression, dict[str, float]]:
        """Train Logistic Regression Model with hyperparameter tuning.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        X_test : pd.DataFrame
            Testing features.
        y_test : pd.Series
            Testing labels.

        Returns
        -------
        tuple[LogisticRegression, dict[str, float]]
        Trained Logistic Regression model and its evaluation metrics.

        """
        logger.info("Training Logistic Regression...")

        with mlflow.start_run(run_name="logistic-regression"):
            param_grid = {"C": [0.001, 0.01, 0.1, 1, 10], "max_iter": [1000]}
            lr = LogisticRegression(random_state=42, solver="lbfgs")
            grid_search = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1, scoring="roc_auc")
            grid_search.fit(X_train, y_train)

            best_lr = grid_search.best_estimator_
            metrics = self._evaluate_model(best_lr, X_train, y_train, X_test, y_test)

            mlflow.log_params({"model": "LogisticRegression", "best_C": grid_search.best_params_["C"]})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_lr, "model")

            self.models["logistic_regression"] = {"model": best_lr, "metrics": metrics}
            logger.info("LR Metrics: %s", metrics)
            return best_lr, metrics

    def train_random_forest(
        self,
        X_train: pd.DataFrame,  # noqa: N803
        y_train: pd.Series,
        X_test: pd.DataFrame,  # noqa: N803
        y_test: pd.Series,
        feature_names: list[str] | None = None,
    ) -> tuple[RandomForestClassifier, dict[str, float]]:
        """Train Random Forest with hyperparameter tuning.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        X_test : pd.DataFrame
            Testing features.
        y_test : pd.Series
            Testing labels.
        feature_names : list[str] | None, optional
            List of feature names for importance plotting, by default None.

        Returns
        -------
        tuple[RandomForestClassifier, dict[str, float]]
            Trained Random Forest model and its evaluation metrics.

        Raises
        ------
        ValueError
            If the length of feature_names does not match the number of features in the model.

        """
        logger.info("Training Random Forest...")

        with mlflow.start_run(run_name="random-forest"):
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5],
            }

            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf,
                param_grid,
                cv=5,
                n_jobs=-1,
                scoring="roc_auc",
            )
            grid_search.fit(X_train, y_train)

            best_rf = grid_search.best_estimator_
            metrics = self._evaluate_model(best_rf, X_train, y_train, X_test, y_test)

            mlflow.log_params(
                {
                    "model": "RandomForest",
                    "best_n_estimators": grid_search.best_params_["n_estimators"],
                    "best_max_depth": grid_search.best_params_["max_depth"],
                    "cv_folds": 5,
                },
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_rf, "model")

            # SAFE FEATURE IMPORTANCE
            if feature_names is None:
                # If X_train is a DataFrame, use its columns; else index features
                if hasattr(X_train, "columns"):
                    feature_names = list(X_train.columns)
                else:
                    feature_names = [f"f{i}" for i in range(len(best_rf.feature_importances_))]

            importances = best_rf.feature_importances_

            # Replace assert with exception for safety
            if len(feature_names) != len(importances):
                msg = f"Feature names length {len(feature_names)} does not match importances length {len(importances)}"
                raise ValueError(
                    msg,
                )

            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": importances},
            ).sort_values("importance", ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=feature_importance.head(10),
                x="importance",
                y="feature",
            )
            plt.title("Random Forest - Top 10 Feature Importance")
            plt.tight_layout()
            Path("logs").mkdir(exist_ok=True)
            plt.savefig("logs/feature_importance.png", dpi=100)
            mlflow.log_artifact("logs/feature_importance.png")
            plt.close()

            self.models["random_forest"] = {
                "model": best_rf,
                "metrics": metrics,
                "best_params": grid_search.best_params_,
            }

            logger.info("RF Metrics: %s", metrics)
            return best_rf, metrics

    @staticmethod
    def _evaluate_model(
        model: BaseEstimator,
        X_train: pd.DataFrame,  # noqa: N803
        y_train: pd.Series,
        X_test: pd.DataFrame,  # noqa: N803
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Evaluate model.

        Parameters
        ----------
        model : BaseEstimator
            Trained model to evaluate.
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        X_test : pd.DataFrame
            Testing features.
        y_test : pd.Series
            Testing labels.

        Returns
        -------
        dict[str, float]
            Evaluation metrics.

        """
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        try:
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
        except Exception:  # noqa: BLE001
            train_auc = 0.0
            test_auc = 0.0

        return {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
            "test_auc": float(test_auc),
            "train_auc": float(train_auc),
        }

    def select_best_model(self) -> BaseEstimator:
        """Select best model based on test AUC.

        Returns
        -------
        BaseEstimator
            The best-performing trained model (based on test AUC).

        """
        best_key = max(self.models.keys(), key=lambda x: self.models[x]["metrics"]["test_auc"])
        self.best_model = self.models[best_key]["model"]
        logger.info("Best model: %s", best_key)
        return self.best_model

    def save_best_model(self, filepath: str = "models/best_model.pkl") -> None:
        """Save best model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with Path(filepath).open("wb") as f:
            pickle.dump(self.best_model, f)
        logger.info("Model saved to %s", filepath)


def train_pipeline(data_path: str = "data/raw/heart_disease_raw.csv") -> tuple[BaseEstimator, HeartDiseasePreprocessor]:
    """Complete training pipeline.

    Parameters
    ----------
    data_path : str, optional
        Path to the raw data CSV file, by default "data/raw/heart_disease_raw.csv"

    Returns
    -------
    tuple[BaseEstimator, HeartDiseasePreprocessor]
        The best trained model and the fitted preprocessor used to transform the data.

    """
    logger.info("Starting training pipeline...")

    # Load data
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data(data_path)  # noqa: N806

    # Preprocess
    preprocessor = HeartDiseasePreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)  # noqa: N806
    X_test_processed = preprocessor.transform(X_test)  # noqa: N806

    # Train models
    trainer = ModelTrainer()
    trainer.train_logistic_regression(X_train_processed, y_train, X_test_processed, y_test)
    trainer.train_random_forest(X_train_processed, y_train, X_test_processed, y_test)

    # Save best
    trainer.select_best_model()
    trainer.save_best_model()
    preprocessor.save("models/preprocessor.pkl")

    logger.info("✓ Training complete!")
    return trainer.best_model, preprocessor
