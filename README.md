# Credit Card Fraud Detection - Research & Education Demo

## DISCLAIMER

**This is a research and education demonstration project. It is NOT intended for production security operations or real-world fraud detection systems.**

- This demo uses synthetic data and simplified models
- Results may be inaccurate and should not be used for actual fraud detection
- This is NOT a SOC (Security Operations Center) tool
- For production fraud detection, consult with security professionals and use certified systems

## Overview

This project demonstrates modern machine learning techniques for credit card fraud detection using synthetic transaction data. It showcases:

- **Imbalanced Learning**: Handling rare fraud events (<1% of transactions)
- **Feature Engineering**: Transaction patterns, behavioral signals, and contextual features
- **Multiple Models**: Gradient boosting, neural networks, and ensemble methods
- **Evaluation Metrics**: AUCPR, precision@K, cost curves, and operational metrics
- **Explainability**: SHAP explanations for fraud reasoning
- **Privacy Safeguards**: Data anonymization and PII protection

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Run the Demo

```bash
# Start the interactive Streamlit demo
streamlit run demo/app.py

# Or run the basic example
python scripts/train_basic.py
```

## Dataset Schema

The synthetic dataset includes the following anonymized features:

- **Transaction Features**: Amount, time, merchant category, location
- **Behavioral Features**: Spending patterns, velocity, frequency
- **Contextual Features**: Device fingerprint, IP geolocation, session data
- **Labels**: Binary fraud indicator (0=legitimate, 1=fraudulent)

All personally identifiable information (PII) has been removed or anonymized.

## Model Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml

# Train with custom parameters
python scripts/train.py --config configs/xgboost.yaml --data.path data/synthetic_fraud.csv
```

## Evaluation

The project includes comprehensive evaluation metrics:

- **Detection Metrics**: AUCPR, precision@K, recall at fixed precision
- **Cost Analysis**: False positive/negative costs, alert workload
- **Operational Metrics**: Detection latency, false alarm rates
- **Robustness**: Cross-validation, temporal splits, entity-aware evaluation

## Demo Features

The Streamlit demo provides:

- **Live Transaction Scoring**: Upload or simulate transactions
- **Fraud Explanation**: SHAP feature importance and reasoning
- **Model Comparison**: Side-by-side performance metrics
- **Threshold Tuning**: Interactive precision/recall optimization
- **Cost Analysis**: Business impact visualization

## Project Structure

```
credit-card-fraud-detection/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # ML models and training
│   ├── evaluation/     # Metrics and evaluation
│   ├── explainability/ # SHAP and model interpretation
│   └── utils/          # Utility functions
├── data/               # Dataset storage
├── configs/            # Configuration files
├── scripts/            # Training and evaluation scripts
├── demo/               # Streamlit demo application
├── tests/              # Unit tests
├── assets/             # Generated plots and results
└── notebooks/          # Jupyter notebooks for exploration
```

## Privacy & Security

- **Data Anonymization**: All PII removed or hashed
- **Synthetic Data**: No real transaction data used
- **Secure Processing**: Input validation and sanitization
- **Audit Logging**: Track all model decisions and explanations

## Limitations

- Uses simplified synthetic data
- Models may not generalize to real-world scenarios
- Does not include all fraud patterns and attack vectors
- Not suitable for production deployment without extensive validation

## Contributing

This is a research demonstration project. For educational purposes only.

## License

MIT License - See LICENSE file for details.
# Credit-Card-Fraud-Detection
