"""Explainability and model interpretation for fraud detection."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig


class FraudExplainer:
    """Model explainability for fraud detection using SHAP."""
    
    def __init__(self, config: DictConfig):
        """Initialize the explainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def fit_explainer(self, model, X_train: pd.DataFrame, X_sample: Optional[pd.DataFrame] = None) -> None:
        """Fit SHAP explainer to the model.
        
        Args:
            model: Trained fraud detection model
            X_train: Training features for background
            X_sample: Sample features for explanation (optional)
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        logger.info("Fitting SHAP explainer")
        
        # Use sample of training data as background
        n_background = min(1000, len(X_train))
        background_indices = np.random.choice(len(X_train), n_background, replace=False)
        background = X_train.iloc[background_indices]
        
        # Create explainer based on model type
        if hasattr(model, 'predict_proba'):
            # For tree-based models, use TreeExplainer
            if hasattr(model.model, 'get_booster'):  # XGBoost
                self.explainer = shap.TreeExplainer(model.model)
            elif hasattr(model.model, 'booster_'):  # LightGBM
                self.explainer = shap.TreeExplainer(model.model)
            else:
                # For other models, use KernelExplainer
                self.explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    background
                )
        else:
            raise ValueError("Model must have predict_proba method")
        
        self.feature_names = X_train.columns.tolist()
        logger.info("SHAP explainer fitted successfully")
    
    def explain_predictions(
        self, 
        model, 
        X: pd.DataFrame, 
        max_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Generate SHAP explanations for predictions.
        
        Args:
            model: Trained fraud detection model
            X: Features to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        logger.info(f"Generating SHAP explanations for {len(X)} samples")
        
        # Limit samples for performance
        if max_samples and len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
        
        # Calculate SHAP values
        if hasattr(self.explainer, 'shap_values'):
            # TreeExplainer
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                # Multi-class case, take fraud class (index 1)
                shap_values = shap_values[1]
        else:
            # KernelExplainer
            shap_values = self.explainer.shap_values(X_sample)
        
        # Get predictions
        predictions = model.predict_proba(X_sample)[:, 1]
        
        # Store results
        self.shap_values = shap_values
        
        return {
            'shap_values': shap_values,
            'predictions': predictions,
            'feature_names': self.feature_names,
            'data': X_sample.values
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from SHAP values.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values available. Run explain_predictions first.")
        
        # Calculate mean absolute SHAP values
        importance_scores = np.mean(np.abs(self.shap_values), axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        logger.info("Generated feature importance from SHAP values")
        return importance_df
    
    def explain_single_prediction(
        self, 
        model, 
        X: pd.DataFrame, 
        sample_idx: int = 0
    ) -> Dict[str, Union[str, float, List]]:
        """Explain a single prediction in detail.
        
        Args:
            model: Trained fraud detection model
            X: Features
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with detailed explanation
        """
        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} out of range")
        
        # Get single sample
        X_sample = X.iloc[[sample_idx]]
        
        # Generate explanation
        explanation = self.explain_predictions(model, X_sample)
        
        # Extract values for single sample
        shap_vals = explanation['shap_values'][0]
        prediction = explanation['predictions'][0]
        feature_values = explanation['data'][0]
        
        # Create detailed explanation
        detailed_explanation = {
            'prediction': prediction,
            'prediction_label': 'Fraud' if prediction > 0.5 else 'Legitimate',
            'confidence': abs(prediction - 0.5) * 2,  # Confidence as distance from 0.5
            'feature_contributions': []
        }
        
        # Sort features by absolute SHAP value
        feature_contributions = list(zip(self.feature_names, shap_vals, feature_values))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Add top contributing features
        top_k = self.config.explainability.feature_importance_top_k
        for feature, shap_val, feature_val in feature_contributions[:top_k]:
            contribution = {
                'feature': feature,
                'value': feature_val,
                'shap_value': shap_val,
                'contribution': shap_val,
                'direction': 'increases' if shap_val > 0 else 'decreases',
                'risk': 'fraud risk' if shap_val > 0 else 'legitimate'
            }
            detailed_explanation['feature_contributions'].append(contribution)
        
        return detailed_explanation
    
    def generate_explanation_report(
        self, 
        model, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> str:
        """Generate a comprehensive explanation report.
        
        Args:
            model: Trained fraud detection model
            X: Features to explain
            y: True labels (optional)
            
        Returns:
            Formatted explanation report
        """
        logger.info("Generating comprehensive explanation report")
        
        # Get explanations
        explanations = self.explain_predictions(model, X)
        
        # Get feature importance
        importance_df = self.get_feature_importance()
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append("FRAUD DETECTION MODEL EXPLANATION REPORT")
        report.append("=" * 60)
        
        # Feature importance
        report.append("\nTOP FEATURE IMPORTANCE:")
        report.append("-" * 25)
        for idx, row in importance_df.head(10).iterrows():
            report.append(f"{row['feature']}: {row['importance']:.4f}")
        
        # Sample explanations
        report.append("\nSAMPLE EXPLANATIONS:")
        report.append("-" * 20)
        
        n_samples = min(3, len(X))
        for i in range(n_samples):
            sample_explanation = self.explain_single_prediction(model, X, i)
            
            report.append(f"\nSample {i+1}:")
            report.append(f"  Prediction: {sample_explanation['prediction_label']} "
                        f"(confidence: {sample_explanation['confidence']:.2f})")
            
            if y is not None:
                actual_label = 'Fraud' if y.iloc[i] == 1 else 'Legitimate'
                report.append(f"  Actual: {actual_label}")
            
            report.append("  Top contributing features:")
            for contrib in sample_explanation['feature_contributions'][:5]:
                report.append(f"    {contrib['feature']}: {contrib['value']:.2f} "
                            f"({contrib['direction']} {contrib['risk']})")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def get_fraud_reasons(
        self, 
        model, 
        X: pd.DataFrame, 
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """Get fraud reasons for high-risk transactions.
        
        Args:
            model: Trained fraud detection model
            X: Features
            threshold: Fraud probability threshold
            
        Returns:
            DataFrame with fraud reasons
        """
        # Get predictions
        predictions = model.predict_proba(X)[:, 1]
        
        # Filter high-risk transactions
        high_risk_mask = predictions >= threshold
        high_risk_indices = np.where(high_risk_mask)[0]
        
        if len(high_risk_indices) == 0:
            logger.warning("No high-risk transactions found")
            return pd.DataFrame()
        
        # Get explanations for high-risk transactions
        high_risk_X = X.iloc[high_risk_indices]
        explanations = self.explain_predictions(model, high_risk_X)
        
        # Extract fraud reasons
        fraud_reasons = []
        for i, idx in enumerate(high_risk_indices):
            shap_vals = explanations['shap_values'][i]
            feature_values = explanations['data'][i]
            
            # Get top risk-increasing features
            risk_features = []
            for j, feature in enumerate(self.feature_names):
                if shap_vals[j] > 0:  # Risk-increasing
                    risk_features.append({
                        'feature': feature,
                        'value': feature_values[j],
                        'risk_contribution': shap_vals[j]
                    })
            
            # Sort by risk contribution
            risk_features.sort(key=lambda x: x['risk_contribution'], reverse=True)
            
            fraud_reasons.append({
                'transaction_id': idx,
                'fraud_probability': predictions[idx],
                'top_risk_factors': risk_features[:5]  # Top 5 risk factors
            })
        
        logger.info(f"Generated fraud reasons for {len(fraud_reasons)} high-risk transactions")
        
        return pd.DataFrame(fraud_reasons)
