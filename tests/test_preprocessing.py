"""Unit tests"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import HeartDiseasePreprocessor


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(29, 78, 100),
            "sex": np.random.randint(0, 2, 100),
            "cp": np.random.randint(0, 4, 100),
            "chol": np.random.randint(126, 565, 100),
            "num": np.random.randint(0, 2, 100),
        }
    )


def test_preprocessor_initialization():
    preprocessor = HeartDiseasePreprocessor()
    assert preprocessor is not None


def test_preprocessor_fit(sample_data):
    X = sample_data.drop("num", axis=1)
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(X)
    assert preprocessor.feature_columns is not None
