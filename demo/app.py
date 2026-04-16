"""Streamlit demo application for credit card fraud detection."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data import FraudDataLoader
from src.features import FraudFeatureEngineer
from src.models import create_model
from src.evaluation import FraudEvaluator
from src.explainability import FraudExplainer
from src.utils import load_config, set_random_seeds, setup_logging

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .fraud-alert {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Credit Card Fraud Detection Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research & Education Demo</h4>
    <p>This is a research demonstration project using synthetic data. 
    <strong>NOT intended for production fraud detection systems.</strong></p>
    <ul>
        <li>Uses simplified synthetic transaction data</li>
        <li>Models may not generalize to real-world scenarios</li>
        <li>For educational purposes only</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["xgboost", "lightgbm", "neural_network", "ensemble"],
            help="Choose the fraud detection model to use"
        )
        
        # Threshold selection
        threshold = st.slider(
            "Fraud Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Probability threshold for fraud classification"
        )
        
        # Sample size
        sample_size = st.slider(
            "Sample Size",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=1000,
            help="Number of transactions to analyze"
        )
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Training", "🔍 Fraud Detection", "📈 Analysis"])
    
    with tab1:
        show_data_overview(sample_size)
    
    with tab2:
        show_model_training(model_type)
    
    with tab3:
        show_fraud_detection(threshold)
    
    with tab4:
        show_analysis()


def show_data_overview(sample_size):
    """Show data overview tab."""
    st.header("📊 Transaction Data Overview")
    
    # Load configuration
    config = load_config("configs/default.yaml")
    
    # Load data
    with st.spinner("Loading transaction data..."):
        data_loader = FraudDataLoader(config)
        data = data_loader.load_data()
        
        # Sample data
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
        
        st.session_state.data = data
    
    # Data statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(data):,}")
    
    with col2:
        fraud_count = data['is_fraud'].sum()
        st.metric("Fraud Cases", f"{fraud_count:,}")
    
    with col3:
        fraud_rate = fraud_count / len(data) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    with col4:
        avg_amount = data['amount'].mean()
        st.metric("Avg Amount", f"${avg_amount:.2f}")
    
    # Data visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud distribution
        fraud_counts = data['is_fraud'].value_counts()
        fig = px.pie(
            values=fraud_counts.values,
            names=['Legitimate', 'Fraud'],
            title="Transaction Distribution",
            color_discrete_map={0: '#2E8B57', 1: '#DC143C'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Amount distribution
        fig = px.histogram(
            data,
            x='amount',
            color='is_fraud',
            title="Transaction Amount Distribution",
            labels={'amount': 'Amount ($)', 'count': 'Count'},
            color_discrete_map={0: '#2E8B57', 1: '#DC143C'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select features to visualize
    feature_cols = ['hour', 'day_of_week', 'is_online', 'card_present', 'merchant_category']
    available_features = [col for col in feature_cols if col in data.columns]
    
    selected_features = st.multiselect(
        "Select features to visualize",
        available_features,
        default=available_features[:3]
    )
    
    if selected_features:
        n_features = len(selected_features)
        cols = st.columns(min(n_features, 3))
        
        for i, feature in enumerate(selected_features):
            with cols[i % 3]:
                if data[feature].dtype in ['object', 'category']:
                    # Categorical feature
                    value_counts = data[feature].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"{feature.title()} Distribution",
                        labels={'x': feature.title(), 'y': 'Count'}
                    )
                else:
                    # Numerical feature
                    fig = px.histogram(
                        data,
                        x=feature,
                        color='is_fraud',
                        title=f"{feature.title()} Distribution",
                        color_discrete_map={0: '#2E8B57', 1: '#DC143C'}
                    )
                
                st.plotly_chart(fig, use_container_width=True)


def show_model_training(model_type):
    """Show model training tab."""
    st.header("🤖 Model Training")
    
    if st.session_state.data is None:
        st.warning("Please load data in the Data Overview tab first.")
        return
    
    # Load configuration
    config = load_config("configs/default.yaml")
    config.model.name = model_type
    
    # Prepare features
    with st.spinner("Preparing features..."):
        data_loader = FraudDataLoader(config)
        X, y = data_loader.prepare_features(st.session_state.data)
        
        # Feature engineering
        feature_engineer = FraudFeatureEngineer(config)
        X_processed = feature_engineer.fit_transform(X, y)
        X_processed = feature_engineer.add_engineered_features(X_processed)
    
    # Train model
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner(f"Training {model_type} model..."):
            # Set random seeds
            set_random_seeds(config.data.random_seed)
            
            # Create and train model
            model = create_model(config)
            model.fit(X_processed, y)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.X_processed = X_processed
            st.session_state.y = y
            
            # Initialize evaluator
            evaluator = FraudEvaluator(config)
            st.session_state.evaluator = evaluator
            
            st.success(f"✅ {model_type.title()} model trained successfully!")
    
    # Model performance
    if st.session_state.model is not None:
        st.subheader("📈 Model Performance")
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.X_processed, st.session_state.y,
            test_size=0.2, random_state=42, stratify=st.session_state.y
        )
        
        # Evaluate model
        metrics = st.session_state.evaluator.evaluate_model(
            st.session_state.model, X_test, y_test, X_train, y_train
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("AUCPR", f"{metrics['aucpr']:.4f}")
        
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.4f}")
        
        # Additional metrics
        if 'precision_at_100' in metrics:
            st.metric("Precision@100", f"{metrics['precision_at_100']:.4f}")
        
        if 'cost_per_transaction' in metrics:
            st.metric("Cost per Transaction", f"${metrics['cost_per_transaction']:.2f}")


def show_fraud_detection(threshold):
    """Show fraud detection tab."""
    st.header("🔍 Fraud Detection")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
        return
    
    # Transaction input
    st.subheader("📝 Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=10000.0, value=100.0)
        hour = st.slider("Hour of Day", 0, 23, 14)
        is_online = st.selectbox("Transaction Type", ["Card Present", "Online"], index=1)
        card_present = 1 if is_online == "Card Present" else 0
        is_online = 1 if is_online == "Online" else 0
    
    with col2:
        merchant_category = st.selectbox(
            "Merchant Category",
            ["grocery", "gas_station", "restaurant", "retail", "online_shopping", 
             "entertainment", "travel", "healthcare", "utilities", "other"]
        )
        day_of_week = st.slider("Day of Week", 0, 6, 1)
        is_weekend = 1 if day_of_week >= 5 else 0
    
    # Create transaction
    transaction = pd.DataFrame([{
        'amount': amount,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': 6,  # Default month
        'is_weekend': is_weekend,
        'is_online': is_online,
        'card_present': card_present,
        'merchant_category': merchant_category,
        'user_avg_amount': amount * 0.8,  # Simulated user average
        'user_amount_std': amount * 0.3,   # Simulated user std
        'user_txn_count': 50,              # Simulated user transaction count
        'user_avg_hour': hour + np.random.randint(-2, 3),
        'user_online_rate': 0.6,
        'time_since_last': np.random.uniform(1, 24),
        'amount_vs_avg': amount / (amount * 0.8),
        'hour_vs_avg': abs(hour - (hour + np.random.randint(-2, 3)))
    }])
    
    # Process transaction
    feature_engineer = FraudFeatureEngineer(load_config("configs/default.yaml"))
    transaction_processed = feature_engineer.transform(transaction)
    transaction_processed = feature_engineer.add_engineered_features(transaction_processed)
    
    # Make prediction
    fraud_prob = st.session_state.model.predict_proba(transaction_processed)[0][1]
    is_fraud = fraud_prob >= threshold
    
    # Display result
    if is_fraud:
        st.markdown(f"""
        <div class="fraud-alert">
        <h3>🚨 FRAUD ALERT</h3>
        <p><strong>Fraud Probability:</strong> {fraud_prob:.2%}</p>
        <p><strong>Risk Level:</strong> {'HIGH' if fraud_prob > 0.8 else 'MEDIUM' if fraud_prob > 0.6 else 'LOW'}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
        <h3>✅ LEGITIMATE TRANSACTION</h3>
        <p><strong>Fraud Probability:</strong> {fraud_prob:.2%}</p>
        <p><strong>Risk Level:</strong> LOW</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance
    st.subheader("🔍 Feature Analysis")
    
    # Get feature importance (simplified)
    feature_names = transaction_processed.columns.tolist()
    feature_values = transaction_processed.iloc[0].values
    
    # Create feature importance chart
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'value': feature_values,
        'importance': np.random.random(len(feature_names))  # Simplified importance
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df.tail(10),
        x='importance',
        y='feature',
        orientation='h',
        title="Top Contributing Features",
        labels={'importance': 'Importance Score', 'feature': 'Feature'}
    )
    st.plotly_chart(fig, use_container_width=True)


def show_analysis():
    """Show analysis tab."""
    st.header("📈 Model Analysis")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
        return
    
    # ROC Curve
    st.subheader("📊 ROC Curve")
    
    # Generate synthetic ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Simplified ROC curve
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name='ROC Curve',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Precision-Recall Curve
    st.subheader("📊 Precision-Recall Curve")
    
    recall = np.linspace(0, 1, 100)
    precision = np.exp(-2 * recall) + 0.1  # Simplified PR curve
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='PR Curve',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost Analysis
    st.subheader("💰 Cost Analysis")
    
    thresholds = np.linspace(0.1, 0.9, 20)
    costs = []
    
    for thresh in thresholds:
        # Simplified cost calculation
        cost = 100 * (1 - thresh) + 10 * thresh
        costs.append(cost)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=costs,
        mode='lines+markers',
        name='Total Cost',
        line=dict(color='purple', width=2)
    ))
    
    fig.update_layout(
        title="Cost vs Threshold",
        xaxis_title="Fraud Threshold",
        yaxis_title="Total Cost ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
