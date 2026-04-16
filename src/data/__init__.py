"""Data loading and generation for credit card fraud detection."""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from ..utils import anonymize_pii, create_directories, validate_dataframe


class FraudDataGenerator:
    """Generate synthetic credit card transaction data for fraud detection."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_transactions(
        self,
        n_transactions: int = 10000,
        fraud_rate: float = 0.01,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31"
    ) -> pd.DataFrame:
        """Generate synthetic credit card transaction data.
        
        Args:
            n_transactions: Total number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions (0.0 to 1.0)
            start_date: Start date for transaction timestamps
            end_date: End date for transaction timestamps
            
        Returns:
            DataFrame with synthetic transaction data
        """
        logger.info(f"Generating {n_transactions:,} transactions with {fraud_rate:.1%} fraud rate")
        
        # Generate base transaction data
        data = self._generate_base_transactions(n_transactions, start_date, end_date)
        
        # Add fraud labels
        n_fraud = int(n_transactions * fraud_rate)
        fraud_indices = np.random.choice(n_transactions, n_fraud, replace=False)
        data['is_fraud'] = 0
        data.loc[fraud_indices, 'is_fraud'] = 1
        
        # Modify fraudulent transactions to have suspicious patterns
        data = self._add_fraud_patterns(data, fraud_indices)
        
        # Add behavioral features
        data = self._add_behavioral_features(data)
        
        # Anonymize PII
        pii_columns = ['card_id', 'merchant_id', 'user_id']
        data = anonymize_pii(data, pii_columns)
        
        logger.info(f"Generated dataset: {len(data):,} transactions, {data['is_fraud'].sum():,} fraud cases")
        
        return data
    
    def _generate_base_transactions(
        self, 
        n_transactions: int, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Generate base transaction features."""
        
        # Time features
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        date_range = (end_dt - start_dt).days
        
        timestamps = [
            start_dt + timedelta(
                days=np.random.randint(0, date_range),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60),
                seconds=np.random.randint(0, 60)
            )
            for _ in range(n_transactions)
        ]
        
        # Transaction amounts (log-normal distribution)
        amounts = np.random.lognormal(mean=3.0, sigma=1.5, size=n_transactions)
        amounts = np.clip(amounts, 1.0, 10000.0)  # Cap at $10,000
        
        # Merchant categories
        merchant_categories = [
            'grocery', 'gas_station', 'restaurant', 'retail', 'online_shopping',
            'entertainment', 'travel', 'healthcare', 'utilities', 'other'
        ]
        
        # Generate base data
        data = pd.DataFrame({
            'timestamp': timestamps,
            'amount': amounts,
            'card_id': [f"card_{i:06d}" for i in range(n_transactions)],
            'merchant_id': [f"merchant_{np.random.randint(1, 1000):04d}" for _ in range(n_transactions)],
            'merchant_category': np.random.choice(merchant_categories, n_transactions),
            'user_id': [f"user_{np.random.randint(1, 500):04d}" for _ in range(n_transactions)],
            'is_online': np.random.choice([0, 1], n_transactions, p=[0.6, 0.4]),
            'card_present': np.random.choice([0, 1], n_transactions, p=[0.3, 0.7]),
        })
        
        # Add derived time features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        return data
    
    def _add_fraud_patterns(self, data: pd.DataFrame, fraud_indices: np.ndarray) -> pd.DataFrame:
        """Add suspicious patterns to fraudulent transactions."""
        
        # High-value transactions
        high_value_mask = np.random.random(len(fraud_indices)) < 0.3
        high_value_indices = fraud_indices[high_value_mask]
        data.loc[high_value_indices, 'amount'] *= np.random.uniform(3, 10, len(high_value_indices))
        
        # Unusual hours (late night/early morning)
        unusual_hour_mask = np.random.random(len(fraud_indices)) < 0.4
        unusual_hour_indices = fraud_indices[unusual_hour_mask]
        data.loc[unusual_hour_indices, 'hour'] = np.random.choice([0, 1, 2, 3, 22, 23], len(unusual_hour_indices))
        
        # Online transactions (higher fraud risk)
        online_mask = np.random.random(len(fraud_indices)) < 0.6
        online_indices = fraud_indices[online_mask]
        data.loc[online_indices, 'is_online'] = 1
        data.loc[online_indices, 'card_present'] = 0
        
        # High-risk merchant categories
        high_risk_categories = ['online_shopping', 'entertainment', 'travel']
        risk_mask = np.random.random(len(fraud_indices)) < 0.5
        risk_indices = fraud_indices[risk_mask]
        data.loc[risk_indices, 'merchant_category'] = np.random.choice(high_risk_categories, len(risk_indices))
        
        return data
    
    def _add_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral features based on user patterns."""
        
        # Calculate user-level statistics
        user_stats = data.groupby('user_id').agg({
            'amount': ['mean', 'std', 'count'],
            'hour': 'mean',
            'is_online': 'mean'
        }).round(2)
        
        user_stats.columns = ['user_avg_amount', 'user_amount_std', 'user_txn_count', 'user_avg_hour', 'user_online_rate']
        
        # Merge user statistics
        data = data.merge(user_stats, left_on='user_id', right_index=True, how='left')
        
        # Add transaction velocity features
        data = data.sort_values(['user_id', 'timestamp'])
        data['time_since_last'] = data.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600  # hours
        data['amount_vs_avg'] = data['amount'] / data['user_avg_amount']
        data['hour_vs_avg'] = abs(data['hour'] - data['user_avg_hour'])
        
        # Fill NaN values
        data['time_since_last'] = data['time_since_last'].fillna(24)  # Default to 24 hours
        data['amount_vs_avg'] = data['amount_vs_avg'].fillna(1.0)
        data['hour_vs_avg'] = data['hour_vs_avg'].fillna(0)
        
        return data


class FraudDataLoader:
    """Load and preprocess fraud detection datasets."""
    
    def __init__(self, config: DictConfig):
        """Initialize the data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_path = Path(config.data.path)
        
    def load_data(self) -> pd.DataFrame:
        """Load transaction data from file or generate if not exists.
        
        Returns:
            DataFrame with transaction data
        """
        if self.data_path.exists():
            logger.info(f"Loading data from: {self.data_path}")
            data = pd.read_csv(self.data_path)
        else:
            logger.info("Data file not found, generating synthetic data")
            generator = FraudDataGenerator(random_seed=self.config.data.random_seed)
            data = generator.generate_transactions(
                n_transactions=50000,
                fraud_rate=0.015,  # 1.5% fraud rate
                start_date="2024-01-01",
                end_date="2024-12-31"
            )
            
            # Save generated data
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(self.data_path, index=False)
            logger.info(f"Saved synthetic data to: {self.data_path}")
        
        # Validate required columns
        required_columns = [
            'timestamp', 'amount', 'hour', 'is_online', 'card_present',
            'merchant_category', 'is_fraud'
        ]
        validate_dataframe(data, required_columns)
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling.
        
        Args:
            data: Raw transaction data
            
        Returns:
            Tuple of (features, target)
        """
        # Select features for modeling
        feature_columns = [
            'amount', 'hour', 'day_of_week', 'month', 'is_weekend',
            'is_online', 'card_present', 'merchant_category',
            'user_avg_amount', 'user_amount_std', 'user_txn_count',
            'user_avg_hour', 'user_online_rate', 'time_since_last',
            'amount_vs_avg', 'hour_vs_avg'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in data.columns]
        X = data[available_features].copy()
        y = data['is_fraud'].copy()
        
        # Handle categorical variables
        categorical_columns = ['merchant_category']
        for col in categorical_columns:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        logger.info(f"Prepared features: {X.shape[1]} features, {len(X):,} samples")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
