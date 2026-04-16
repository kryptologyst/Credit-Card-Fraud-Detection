#!/usr/bin/env python3
"""Basic training script for quick testing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data import FraudDataLoader
from src.features import FraudFeatureEngineer
from src.models import create_model
from src.evaluation import FraudEvaluator
from src.utils import load_config, set_random_seeds, setup_logging


def main():
    """Quick training example."""
    print("Credit Card Fraud Detection - Quick Training")
    print("=" * 50)
    
    # Load configuration
    config = load_config("configs/default.yaml")
    config.model.name = "xgboost"
    
    # Setup
    set_random_seeds(42)
    setup_logging("INFO")
    
    # Load and prepare data
    print("Loading data...")
    data_loader = FraudDataLoader(config)
    data = data_loader.load_data()
    
    # Use smaller sample for quick training
    data = data.sample(n=min(5000, len(data)), random_state=42)
    
    X, y = data_loader.prepare_features(data)
    
    # Feature engineering
    feature_engineer = FraudFeatureEngineer(config)
    X_processed = feature_engineer.fit_transform(X, y)
    X_processed = feature_engineer.add_engineered_features(X_processed)
    
    print(f"Features: {X_processed.shape}")
    print(f"Fraud rate: {y.mean():.2%}")
    
    # Train model
    print("Training XGBoost model...")
    model = create_model(config)
    model.fit(X_processed, y)
    
    # Evaluate
    evaluator = FraudEvaluator(config)
    metrics = evaluator.evaluate_model(model, X_processed, y)
    
    print("\nResults:")
    print(f"AUCPR: {metrics['aucpr']:.4f}")
    print(f"AUC:   {metrics['auc']:.4f}")
    print(f"F1:    {metrics['f1']:.4f}")
    
    # Test prediction
    print("\nTesting prediction...")
    test_transaction = X_processed.iloc[0:1]
    fraud_prob = model.predict_proba(test_transaction)[0][1]
    print(f"Sample fraud probability: {fraud_prob:.2%}")
    
    print("\n✅ Quick training completed!")


if __name__ == "__main__":
    main()
