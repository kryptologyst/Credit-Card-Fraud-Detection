"""Basic tests for the fraud detection system."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import FraudDataGenerator, FraudDataLoader
from src.features import FraudFeatureEngineer
from src.models import create_model
from src.evaluation import FraudEvaluator
from src.utils import load_config, set_random_seeds


class TestFraudDetection:
    """Test cases for fraud detection system."""
    
    def setup_method(self):
        """Setup test data and configuration."""
        self.config = load_config("configs/default.yaml")
        self.config.model.name = "xgboost"
        set_random_seeds(42)
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        generator = FraudDataGenerator(random_seed=42)
        data = generator.generate_transactions(n_transactions=1000, fraud_rate=0.05)
        
        assert len(data) == 1000
        assert 'is_fraud' in data.columns
        assert data['is_fraud'].sum() == 50  # 5% of 1000
        assert 'amount' in data.columns
        assert 'hour' in data.columns
        assert 'is_online' in data.columns
    
    def test_data_loading(self):
        """Test data loading functionality."""
        data_loader = FraudDataLoader(self.config)
        data = data_loader.load_data()
        
        assert len(data) > 0
        assert 'is_fraud' in data.columns
        assert data['is_fraud'].dtype in [int, bool]
    
    def test_feature_engineering(self):
        """Test feature engineering pipeline."""
        # Generate test data
        generator = FraudDataGenerator(random_seed=42)
        data = generator.generate_transactions(n_transactions=500, fraud_rate=0.1)
        
        # Prepare features
        data_loader = FraudDataLoader(self.config)
        X, y = data_loader.prepare_features(data)
        
        # Feature engineering
        feature_engineer = FraudFeatureEngineer(self.config)
        X_processed = feature_engineer.fit_transform(X, y)
        X_processed = feature_engineer.add_engineered_features(X_processed)
        
        assert X_processed.shape[0] == len(data)
        assert X_processed.shape[1] > X.shape[1]  # Should have more features
        assert not X_processed.isnull().any().any()  # No NaN values
    
    def test_model_training(self):
        """Test model training."""
        # Generate test data
        generator = FraudDataGenerator(random_seed=42)
        data = generator.generate_transactions(n_transactions=1000, fraud_rate=0.1)
        
        # Prepare features
        data_loader = FraudDataLoader(self.config)
        X, y = data_loader.prepare_features(data)
        
        feature_engineer = FraudFeatureEngineer(self.config)
        X_processed = feature_engineer.fit_transform(X, y)
        
        # Train model
        model = create_model(self.config)
        model.fit(X_processed, y)
        
        assert model.is_fitted
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Generate test data
        generator = FraudDataGenerator(random_seed=42)
        data = generator.generate_transactions(n_transactions=500, fraud_rate=0.1)
        
        # Prepare features
        data_loader = FraudDataLoader(self.config)
        X, y = data_loader.prepare_features(data)
        
        feature_engineer = FraudFeatureEngineer(self.config)
        X_processed = feature_engineer.fit_transform(X, y)
        
        # Train model
        model = create_model(self.config)
        model.fit(X_processed, y)
        
        # Test predictions
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)
        
        assert len(predictions) == len(X_processed)
        assert len(probabilities) == len(X_processed)
        assert probabilities.shape[1] == 2  # Binary classification
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0 <= prob <= 1 for prob in probabilities.flatten())
    
    def test_evaluation(self):
        """Test evaluation metrics."""
        # Generate test data
        generator = FraudDataGenerator(random_seed=42)
        data = generator.generate_transactions(n_transactions=1000, fraud_rate=0.1)
        
        # Prepare features
        data_loader = FraudDataLoader(self.config)
        X, y = data_loader.prepare_features(data)
        
        feature_engineer = FraudFeatureEngineer(self.config)
        X_processed = feature_engineer.fit_transform(X, y)
        
        # Train model
        model = create_model(self.config)
        model.fit(X_processed, y)
        
        # Evaluate
        evaluator = FraudEvaluator(self.config)
        metrics = evaluator.evaluate_model(model, X_processed, y)
        
        assert 'aucpr' in metrics
        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['aucpr'] <= 1
        assert 0 <= metrics['auc'] <= 1
    
    def test_explainability(self):
        """Test model explainability."""
        # Generate test data
        generator = FraudDataGenerator(random_seed=42)
        data = generator.generate_transactions(n_transactions=500, fraud_rate=0.1)
        
        # Prepare features
        data_loader = FraudDataLoader(self.config)
        X, y = data_loader.prepare_features(data)
        
        feature_engineer = FraudFeatureEngineer(self.config)
        X_processed = feature_engineer.fit_transform(X, y)
        
        # Train model
        model = create_model(self.config)
        model.fit(X_processed, y)
        
        # Test explainability (without SHAP to avoid dependency issues)
        explainer = FraudExplainer(self.config)
        
        # This test just ensures the explainer can be created
        assert explainer is not None
        assert explainer.config == self.config


if __name__ == "__main__":
    pytest.main([__file__])
