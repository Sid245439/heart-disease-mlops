"""
Data Processing Pipeline for Heart Disease
Robust NaN handling + binary target conversion
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeartDiseasePreprocessor:
    """Preprocessing pipeline - ROBUST NaN handling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_columns = None
        self.numeric_features = None
        self.categorical_features = None
        self.numeric_medians = None  # FIXED: PLURAL
        
    def fit(self, X):
        """Fit on training data"""
        self.feature_columns = X.columns.tolist()
        self.numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features = [col for col in X.columns if col not in self.numeric_features]
        
        # Store medians for NaN filling
        if self.numeric_features:
            self.numeric_medians = X[self.numeric_features].median()
            X_numeric_filled = X[self.numeric_features].fillna(self.numeric_medians)
            self.scaler.fit(X_numeric_filled)
        
        # Fit encoders
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna('missing'))
            self.encoders[col] = le
            
        logger.info(f"✓ Fitted. Numeric: {len(self.numeric_features)}, Categorical: {len(self.categorical_features)}")
        return self
    
    def transform(self, X):
        """Transform with NaN safety"""
        X = X.copy()
        
        # Fix column order
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = np.nan
            X = X.reindex(columns=self.feature_columns)
        
        # Fill numeric NaNs
        if self.numeric_features and self.numeric_medians is not None:
            X[self.numeric_features] = X[self.numeric_features].fillna(self.numeric_medians)
        
        # Scale numerics
        if self.numeric_features:
            X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        
        # Encode categoricals
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = self.encoders[col].transform(X[col].astype(str).fillna('missing'))
        
        # FINAL NaN CHECK
        if X.isna().any().any():
            raise ValueError(f"Output contains NaNs: {X.isna().sum().sum()}")
            
        return X
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved: {filepath}")
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def load_and_prepare_data(raw_path, test_size=0.2, random_state=42):
    """Load + create binary target + split"""
    logger.info(f"Loading: {raw_path}")
    
    df = pd.read_csv(raw_path)
    orig_target = df.columns[-1]
    
    logger.info(f"Original target: {df[orig_target].value_counts().to_dict()}")
    
    # Binary target: 0=no disease, 1=disease
    df["target_binary"] = (df[orig_target] > 0).astype(int)
    
    # X = all except BOTH targets
    X = df.drop(columns=[orig_target, "target_binary"])
    y = df["target_binary"]
    
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()
