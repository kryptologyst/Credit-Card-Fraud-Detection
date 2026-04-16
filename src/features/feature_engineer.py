"""Feature engineering for credit card fraud detection."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


class FraudFeatureEngineer:
    """Feature engineering pipeline for fraud detection."""
    
    def __init__(self, config: DictConfig):
        """Initialize the feature engineer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.preprocessor = None
        self.feature_selector = None
        self.label_encoders = {}
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the preprocessing pipeline and transform features.
        
        Args:
            X: Input features
            y: Target labels (optional, needed for feature selection)
            
        Returns:
            Transformed features
        """
        logger.info("Fitting feature engineering pipeline")
        
        # Identify column types
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['category', 'object']).columns.tolist()
        
        logger.info(f"Numerical columns: {numerical_columns}")
        logger.info(f"Categorical columns: {categorical_columns}")
        
        # Create preprocessing pipeline
        self.preprocessor = self._create_preprocessor(numerical_columns, categorical_columns)
        
        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        feature_names = self._get_feature_names(numerical_columns, categorical_columns)
        
        # Create DataFrame with proper column names
        X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        
        # Apply feature selection if enabled
        if self.config.features.feature_selection and y is not None:
            X_df = self._apply_feature_selection(X_df, y)
        
        logger.info(f"Feature engineering complete: {X_df.shape[1]} features")
        
        return X_df
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        # Transform features
        X_transformed = self.preprocessor.transform(X)
        
        # Get feature names
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['category', 'object']).columns.tolist()
        feature_names = self._get_feature_names(numerical_columns, categorical_columns)
        
        # Create DataFrame
        X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        
        # Apply feature selection if enabled
        if self.feature_selector is not None:
            X_df = pd.DataFrame(
                self.feature_selector.transform(X_df),
                columns=X_df.columns[self.feature_selector.get_support()],
                index=X_df.index
            )
        
        return X_df
    
    def _create_preprocessor(
        self, 
        numerical_columns: List[str], 
        categorical_columns: List[str]
    ) -> ColumnTransformer:
        """Create the preprocessing pipeline.
        
        Args:
            numerical_columns: List of numerical column names
            categorical_columns: List of categorical column names
            
        Returns:
            Fitted ColumnTransformer
        """
        transformers = []
        
        # Numerical preprocessing
        if numerical_columns:
            scaling_method = self.config.features.numerical_scaling
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            else:
                scaler = "passthrough"
            
            transformers.append(("num", scaler, numerical_columns))
        
        # Categorical preprocessing
        if categorical_columns:
            encoding_method = self.config.features.categorical_encoding
            if encoding_method == "onehot":
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            elif encoding_method == "target":
                # Use target encoding (simplified version)
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                encoder = "passthrough"
            
            transformers.append(("cat", encoder, categorical_columns))
        
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
    
    def _get_feature_names(
        self, 
        numerical_columns: List[str], 
        categorical_columns: List[str]
    ) -> List[str]:
        """Get feature names after preprocessing.
        
        Args:
            numerical_columns: List of numerical column names
            categorical_columns: List of categorical column names
            
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Add numerical feature names
        feature_names.extend(numerical_columns)
        
        # Add categorical feature names
        if categorical_columns and self.preprocessor is not None:
            cat_transformer = self.preprocessor.named_transformers_["cat"]
            if hasattr(cat_transformer, "get_feature_names_out"):
                cat_features = cat_transformer.get_feature_names_out(categorical_columns)
                feature_names.extend(cat_features)
            else:
                # Fallback for simple encoders
                feature_names.extend(categorical_columns)
        
        return feature_names
    
    def _apply_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply feature selection.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Features after selection
        """
        max_features = min(
            self.config.features.max_features,
            X.shape[1]
        )
        
        self.feature_selector = SelectKBest(
            score_func=f_classif,
            k=max_features
        )
        
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def add_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for fraud detection.
        
        Args:
            X: Input features
            
        Returns:
            Features with additional engineered columns
        """
        X_eng = X.copy()
        
        # Amount-based features
        if 'amount' in X_eng.columns:
            X_eng['amount_log'] = np.log1p(X_eng['amount'])
            X_eng['amount_sqrt'] = np.sqrt(X_eng['amount'])
            X_eng['amount_high'] = (X_eng['amount'] > X_eng['amount'].quantile(0.95)).astype(int)
        
        # Time-based features
        if 'hour' in X_eng.columns:
            X_eng['hour_sin'] = np.sin(2 * np.pi * X_eng['hour'] / 24)
            X_eng['hour_cos'] = np.cos(2 * np.pi * X_eng['hour'] / 24)
            X_eng['is_night'] = ((X_eng['hour'] >= 22) | (X_eng['hour'] <= 6)).astype(int)
            X_eng['is_morning'] = ((X_eng['hour'] >= 6) & (X_eng['hour'] <= 12)).astype(int)
            X_eng['is_afternoon'] = ((X_eng['hour'] >= 12) & (X_eng['hour'] <= 18)).astype(int)
            X_eng['is_evening'] = ((X_eng['hour'] >= 18) & (X_eng['hour'] <= 22)).astype(int)
        
        # Interaction features
        if 'is_online' in X_eng.columns and 'amount' in X_eng.columns:
            X_eng['online_amount'] = X_eng['is_online'] * X_eng['amount']
        
        if 'is_weekend' in X_eng.columns and 'hour' in X_eng.columns:
            X_eng['weekend_hour'] = X_eng['is_weekend'] * X_eng['hour']
        
        # Risk score features
        risk_features = []
        if 'is_online' in X_eng.columns:
            risk_features.append('is_online')
        if 'is_weekend' in X_eng.columns:
            risk_features.append('is_weekend')
        if 'is_night' in X_eng.columns:
            risk_features.append('is_night')
        
        if risk_features:
            X_eng['risk_score'] = X_eng[risk_features].sum(axis=1)
        
        logger.info(f"Added {X_eng.shape[1] - X.shape[1]} engineered features")
        
        return X_eng
