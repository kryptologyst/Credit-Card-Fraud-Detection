"""Evaluation metrics and utilities for fraud detection."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class FraudEvaluator:
    """Comprehensive evaluation for fraud detection models."""
    
    def __init__(self, config: DictConfig):
        """Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.metrics_history = []
        
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Evaluate a fraud detection model comprehensively.
        
        Args:
            model: Trained fraud detection model
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional, for cross-validation)
            y_train: Training labels (optional, for cross-validation)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating fraud detection model")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._calculate_basic_metrics(y_test, y_pred, y_pred_proba))
        
        # Fraud-specific metrics
        metrics.update(self._calculate_fraud_metrics(y_test, y_pred, y_pred_proba))
        
        # Cost-based metrics
        metrics.update(self._calculate_cost_metrics(y_test, y_pred, y_pred_proba))
        
        # Cross-validation if training data provided
        if X_train is not None and y_train is not None:
            cv_metrics = self._cross_validate_model(model, X_train, y_train)
            metrics.update(cv_metrics)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        logger.info(f"Evaluation completed. AUCPR: {metrics.get('aucpr', 0):.4f}")
        
        return metrics
    
    def _calculate_basic_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'aucpr': self._calculate_aucpr(y_true, y_pred_proba)
        }
    
    def _calculate_fraud_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate fraud-specific metrics."""
        metrics = {}
        
        # Precision@K metrics
        k_values = self.config.evaluation.k_values
        for k in k_values:
            if len(y_true) >= k:
                metrics[f'precision_at_{k}'] = self._calculate_precision_at_k(
                    y_true, y_pred_proba, k
                )
        
        # Recall at fixed precision
        target_precision = self.config.evaluation.target_precision
        metrics[f'recall_at_precision_{target_precision}'] = self._calculate_recall_at_precision(
            y_true, y_pred_proba, target_precision
        )
        
        # FPR at target TPR
        target_tpr = self.config.evaluation.target_tpr
        metrics[f'fpr_at_tpr_{target_tpr}'] = self._calculate_fpr_at_tpr(
            y_true, y_pred_proba, target_tpr
        )
        
        return metrics
    
    def _calculate_cost_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate cost-based metrics."""
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Cost assumptions (can be configured)
        cost_fp = 1.0  # Cost of false positive (investigation cost)
        cost_fn = 10.0  # Cost of false negative (fraud loss)
        
        # Total cost
        total_cost = fp * cost_fp + fn * cost_fn
        
        # Cost per transaction
        cost_per_transaction = total_cost / len(y_true)
        
        # Cost curve metrics
        thresholds = np.linspace(0, 1, 100)
        costs = []
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_thresh).ravel()
            cost = fp_t * cost_fp + fn_t * cost_fn
            costs.append(cost)
        
        min_cost = min(costs)
        min_cost_threshold = thresholds[np.argmin(costs)]
        
        return {
            'total_cost': total_cost,
            'cost_per_transaction': cost_per_transaction,
            'min_cost': min_cost,
            'min_cost_threshold': min_cost_threshold,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
    
    def _calculate_aucpr(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate Area Under Precision-Recall Curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        # Calculate AUC using trapezoidal rule
        aucpr = np.trapz(precision, recall)
        return aucpr
    
    def _calculate_precision_at_k(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray, 
        k: int
    ) -> float:
        """Calculate precision@K."""
        # Get top K predictions
        top_k_indices = np.argsort(y_pred_proba)[-k:]
        top_k_labels = y_true.iloc[top_k_indices]
        
        return precision_score([1] * k, top_k_labels, zero_division=0)
    
    def _calculate_recall_at_precision(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray, 
        target_precision: float
    ) -> float:
        """Calculate recall at fixed precision."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find threshold that achieves target precision
        valid_indices = precision >= target_precision
        if not np.any(valid_indices):
            return 0.0
        
        # Get maximum recall at target precision
        max_recall = np.max(recall[valid_indices])
        return max_recall
    
    def _calculate_fpr_at_tpr(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray, 
        target_tpr: float
    ) -> float:
        """Calculate FPR at target TPR."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Find threshold that achieves target TPR
        valid_indices = tpr >= target_tpr
        if not np.any(valid_indices):
            return 1.0
        
        # Get minimum FPR at target TPR
        min_fpr = np.min(fpr[valid_indices])
        return min_fpr
    
    def _cross_validate_model(
        self, 
        model, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Dict[str, float]:
        """Perform cross-validation."""
        from sklearn.model_selection import StratifiedKFold
        
        cv_folds = self.config.evaluation.cv_folds
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model on fold
            fold_model = model.__class__(self.config)
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Evaluate on validation set
            y_fold_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
            fold_aucpr = self._calculate_aucpr(y_fold_val, y_fold_pred_proba)
            cv_scores.append(fold_aucpr)
        
        return {
            'cv_aucpr_mean': np.mean(cv_scores),
            'cv_aucpr_std': np.std(cv_scores),
            'cv_aucpr_min': np.min(cv_scores),
            'cv_aucpr_max': np.max(cv_scores)
        }
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard from evaluation history.
        
        Returns:
            DataFrame with model performance comparison
        """
        if not self.metrics_history:
            logger.warning("No evaluation history found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        leaderboard = pd.DataFrame(self.metrics_history)
        
        # Add model names if available
        if 'model_name' not in leaderboard.columns:
            leaderboard['model_name'] = [f"Model_{i+1}" for i in range(len(leaderboard))]
        
        # Sort by AUCPR (primary metric for fraud detection)
        if 'aucpr' in leaderboard.columns:
            leaderboard = leaderboard.sort_values('aucpr', ascending=False)
        
        logger.info("Created evaluation leaderboard")
        return leaderboard
    
    def generate_report(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append("=" * 60)
        report.append("FRAUD DETECTION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(y_true, y_pred, y_pred_proba)
        report.append("\nBASIC METRICS:")
        report.append("-" * 20)
        for metric, value in basic_metrics.items():
            report.append(f"{metric.upper()}: {value:.4f}")
        
        # Fraud-specific metrics
        fraud_metrics = self._calculate_fraud_metrics(y_true, y_pred, y_pred_proba)
        report.append("\nFRAUD-SPECIFIC METRICS:")
        report.append("-" * 25)
        for metric, value in fraud_metrics.items():
            report.append(f"{metric.upper()}: {value:.4f}")
        
        # Cost metrics
        cost_metrics = self._calculate_cost_metrics(y_true, y_pred, y_pred_proba)
        report.append("\nCOST METRICS:")
        report.append("-" * 15)
        for metric, value in cost_metrics.items():
            if 'rate' in metric:
                report.append(f"{metric.upper()}: {value:.4f}")
            else:
                report.append(f"{metric.upper()}: ${value:.2f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        report.append("\nCONFUSION MATRIX:")
        report.append("-" * 18)
        report.append(f"True Negatives:  {cm[0,0]:,}")
        report.append(f"False Positives: {cm[0,1]:,}")
        report.append(f"False Negatives: {cm[1,0]:,}")
        report.append(f"True Positives:  {cm[1,1]:,}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
