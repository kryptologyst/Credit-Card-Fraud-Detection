"""Credit Card Fraud Detection - Research & Education Demo

A comprehensive fraud detection system using modern machine learning techniques.
This is a research demonstration project, not intended for production use.
"""

from .data import FraudDataGenerator, FraudDataLoader
from .evaluation import FraudEvaluator
from .explainability import FraudExplainer
from .features import FraudFeatureEngineer
from .models import (
    BaseFraudModel,
    EnsembleFraudModel,
    LightGBMFraudModel,
    NeuralNetworkFraudModel,
    XGBoostFraudModel,
    create_model,
)
from .utils import (
    anonymize_pii,
    calculate_class_weights,
    create_directories,
    format_currency,
    format_percentage,
    get_device,
    load_config,
    safe_divide,
    save_config,
    set_random_seeds,
    setup_logging,
    validate_dataframe,
)

__version__ = "1.0.0"
__author__ = "Security Research Team"

__all__ = [
    # Data
    "FraudDataGenerator",
    "FraudDataLoader",
    # Features
    "FraudFeatureEngineer",
    # Models
    "BaseFraudModel",
    "XGBoostFraudModel",
    "LightGBMFraudModel", 
    "NeuralNetworkFraudModel",
    "EnsembleFraudModel",
    "create_model",
    # Evaluation
    "FraudEvaluator",
    # Explainability
    "FraudExplainer",
    # Utils
    "anonymize_pii",
    "calculate_class_weights",
    "create_directories",
    "format_currency",
    "format_percentage",
    "get_device",
    "load_config",
    "safe_divide",
    "save_config",
    "set_random_seeds",
    "setup_logging",
    "validate_dataframe",
]
