#!/usr/bin/env python3
"""Training script for credit card fraud detection models."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data import FraudDataLoader
from src.features import FraudFeatureEngineer
from src.models import create_model
from src.evaluation import FraudEvaluator
from src.explainability import FraudExplainer
from src.utils import load_config, set_random_seeds, setup_logging, create_directories


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "lightgbm", "neural_network", "ensemble"],
        default="xgboost",
        help="Model type to train"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/models",
        help="Output directory for trained models"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config.model.name = args.model
    
    # Setup logging
    setup_logging(
        log_level=config.logging.level,
        log_file=config.logging.file
    )
    
    # Set random seeds
    set_random_seeds(config.data.random_seed)
    
    # Create output directories
    create_directories([args.output_dir, "assets/plots", "logs"])
    
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    data_loader = FraudDataLoader(config)
    data = data_loader.load_data()
    
    # Prepare features
    print("🔧 Preparing features...")
    X, y = data_loader.prepare_features(data)
    
    # Feature engineering
    feature_engineer = FraudFeatureEngineer(config)
    X_processed = feature_engineer.fit_transform(X, y)
    X_processed = feature_engineer.add_engineered_features(X_processed)
    
    print(f"Features shape: {X_processed.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=config.data.test_size,
        random_state=config.data.random_seed,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Train model
    print(f"\n🤖 Training {args.model} model...")
    model = create_model(config)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    evaluator = FraudEvaluator(config)
    metrics = evaluator.evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Print results
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
    
    # Explain sample predictions
    sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
    sample_X = X_test.iloc[sample_indices]
    sample_y = y_test.iloc[sample_indices]
    
    explanations = explainer.explain_predictions(model, sample_X)
    
    # Feature importance
    importance_df = explainer.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:<25} {row['importance']:.4f}")
    
    # Save model and results
    print(f"\n💾 Saving model to {args.output_dir}...")
    
    import joblib
    model_path = Path(args.output_dir) / f"{args.model}_fraud_model.pkl"
    joblib.dump(model, model_path)
    
    # Save feature engineer
    feature_path = Path(args.output_dir) / "feature_engineer.pkl"
    joblib.dump(feature_engineer, feature_path)
    
    # Save explainer
    explainer_path = Path(args.output_dir) / "explainer.pkl"
    joblib.dump(explainer, explainer_path)
    
    # Save metrics
    import json
    metrics_path = Path(args.output_dir) / f"{args.model}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature importance
    importance_path = Path(args.output_dir) / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    
    print(f"✅ Training completed successfully!")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Feature importance saved: {importance_path}")


if __name__ == "__main__":
    import numpy as np
    main()
