"""Utility functions for credit card fraud detection."""

import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def anonymize_pii(data: pd.DataFrame, pii_columns: List[str]) -> pd.DataFrame:
    """Anonymize personally identifiable information by hashing.
    
    Args:
        data: DataFrame containing PII
        pii_columns: List of column names containing PII
        
    Returns:
        DataFrame with anonymized PII
    """
    data_anon = data.copy()
    
    for col in pii_columns:
        if col in data_anon.columns:
            # Hash PII with salt for consistency
            salt = "fraud_detection_salt_2024"
            data_anon[col] = data_anon[col].astype(str).apply(
                lambda x: hashlib.sha256((x + salt).encode()).hexdigest()[:16]
            )
            logger.info(f"Anonymized PII column: {col}")
    
    return data_anon


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded configuration from: {config_path}")
    return config


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)
    
    logger.info(f"Saved configuration to: {output_path}")


def create_directories(paths: List[Union[str, Path]]) -> None:
    """Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y
    )
    
    class_weights = dict(zip(classes, weights))
    logger.info(f"Calculated class weights: {class_weights}")
    
    return class_weights


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string.
    
    Args:
        value: Value to format (0.0 to 1.0)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format a float as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"
