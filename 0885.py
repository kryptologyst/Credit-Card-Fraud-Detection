#!/usr/bin/env python3
"""Modernized Credit Card Fraud Detection - Project 885

This is a comprehensive fraud detection system using modern machine learning techniques.
This is a research demonstration project, not intended for production use.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data import FraudDataLoader
from src.features import FraudFeatureEngineer
from src.models import create_model
from src.evaluation import FraudEvaluator
from src.explainability import FraudExplainer
from src.utils import load_config, set_random_seeds, setup_logging


def main():
    """Main function demonstrating the modernized fraud detection system."""
    
    print("=" * 80)
    print("CREDIT CARD FRAUD DETECTION - MODERNIZED SYSTEM")
    print("=" * 80)
    print("Research & Education Demo - NOT for production use")
    print("=" * 80)
    
    # Load configuration
    config = load_config("configs/default.yaml")
    config.model.name = "xgboost"  # Use XGBoost for this demo
    
    # Setup logging and random seeds
    setup_logging("INFO")
    set_random_seeds(config.data.random_seed)
    
    print("\n📊 Loading and preparing data...")
    
    # Load data
    data_loader = FraudDataLoader(config)
    data = data_loader.load_data()
    
    # Use a sample for this demo
    data = data.sample(n=min(10000, len(data)), random_state=42)
    
    # Prepare features
    X, y = data_loader.prepare_features(data)
    
    # Feature engineering
    feature_engineer = FraudFeatureEngineer(config)
    X_processed = feature_engineer.fit_transform(X, y)
    X_processed = feature_engineer.add_engineered_features(X_processed)
    
    print(f"Dataset: {len(data):,} transactions")
    print(f"Features: {X_processed.shape[1]} engineered features")
    print(f"Fraud rate: {y.mean():.2%}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Train model
    print(f"\n🤖 Training {config.model.name} model...")
    model = create_model(config)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\n📈 Evaluating model performance...")
    evaluator = FraudEvaluator(config)
    metrics = evaluator.evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"AUCPR:           {metrics['aucpr']:.4f}")
    print(f"AUC:             {metrics['auc']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"F1 Score:        {metrics['f1']:.4f}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    
    if 'precision_at_100' in metrics:
        print(f"Precision@100:   {metrics['precision_at_100']:.4f}")
    
    if 'cost_per_transaction' in metrics:
        print(f"Cost/Transaction: ${metrics['cost_per_transaction']:.2f}")
    
    print("=" * 60)
    
    # Generate explanations
    print("\n🔍 Generating model explanations...")
    explainer = FraudExplainer(config)
    explainer.fit_explainer(model, X_train)
    
    # Get feature importance
    importance_df = explainer.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    # Test predictions on sample transactions
    print("\n🎯 Testing predictions on sample transactions...")
    
    # Sample some test transactions
    sample_indices = [0, 1, 2] if len(X_test) >= 3 else [0]
    for i, idx in enumerate(sample_indices):
        if idx < len(X_test):
            transaction = X_test.iloc[[idx]]
            fraud_prob = model.predict_proba(transaction)[0][1]
            actual_label = "Fraud" if y_test.iloc[idx] == 1 else "Legitimate"
            
            print(f"\nTransaction {i+1}:")
            print(f"  Amount: ${data.iloc[X_test.index[idx]]['amount']:.2f}")
            print(f"  Hour: {data.iloc[X_test.index[idx]]['hour']}")
            print(f"  Online: {'Yes' if data.iloc[X_test.index[idx]]['is_online'] else 'No'}")
            print(f"  Predicted Fraud Risk: {fraud_prob:.2%}")
            print(f"  Actual Label: {actual_label}")
            print(f"  Prediction: {'FRAUD ALERT' if fraud_prob > 0.5 else 'LEGITIMATE'}")
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive evaluation report...")
    report = evaluator.generate_report(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    print(report)
    
    print("\n" + "=" * 80)
    print("✅ MODERNIZED FRAUD DETECTION DEMO COMPLETED")
    print("=" * 80)
    print("This was a research demonstration using synthetic data.")
    print("For production fraud detection, consult security professionals.")
    print("=" * 80)


if __name__ == "__main__":
    main()