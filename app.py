import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and required files
@st.cache_resource
def load_resources():
    model = joblib.load('fraud_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_info = joblib.load('model_info.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, model_info, feature_names

# Function to make predictions
def predict_fraud(data, model, scaler):
    # Scale the data
    scaled_data = scaler.transform(data)
    # Get prediction and probability
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[:, 1]
    return prediction, probability

# Check if model files exist
if not os.path.exists('fraud_detection_model.pkl'):
    st.error("Model files not found. Please run the training script first.")
    st.stop()

# Load required resources
model, scaler, model_info, feature_names = load_resources()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-card-front-side.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Single Prediction", "Batch Prediction", "Model Information"])

# Display model info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write(f"Model Type: {model_info['model_name']}")
st.sidebar.write(f"AUC Score: {model_info['auc_score']:.4f}")
st.sidebar.write(f"Last Training: {model_info['training_date']}")

# Home page
if page == "Home":
    st.title("Credit Card Fraud Detection System")
    st.markdown("""
    ### Welcome to the Credit Card Fraud Detection System
    
    This application uses machine learning to detect fraudulent credit card transactions.
    
    #### Features:
    - **Single Transaction Analysis**: Check if a specific transaction is likely fraudulent
    - **Batch Processing**: Upload multiple transactions for analysis
    - **Model Information**: View details about the trained model
    
    #### How it works:
    Our system uses a {model_info['model_name']} model trained on thousands of transactions 
    to identify patterns associated with fraud. The model analyzes transaction features 
    and assigns a fraud probability score to each transaction.
    
    Get started by selecting an option from the sidebar!
    """)
    
    # Sample images of dashboard components
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud Detection in Action")
        st.image("random_forest_confusion_matrix.png", caption="Model Performance Visualization")
    
    with col2:
        st.subheader("How to Use This Tool")
        st.markdown("""
        1. Navigate to "Single Prediction" to analyze one transaction
        2. Use "Batch Prediction" to process multiple transactions at once
        3. Check "Model Information" to learn about the underlying system
        
        The system will provide a fraud probability score and classification for each transaction.
        """)

# Single prediction page
elif page == "Single Prediction":
    st.title("Analyze Single Transaction")
    st.write("Enter transaction details to check for potential fraud.")
    
    # Create form for transaction details
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time (seconds from first transaction)", min_value=0, value=0)
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.00, step=10.0)
        
        with col2:
            st.markdown("#### Advanced Features")
            st.write("You can leave these at default values if unknown.")
            
            # Create 4 columns for V1-V28 features to save space
            v_values = []
            num_v_features = 28
            cols_per_row = 4
            num_rows = num_v_features // cols_per_row + (1 if num_v_features % cols_per_row > 0 else 0)
            
            for row in range(num_rows):
                v_cols = st.columns(cols_per_row)
                for col in range(cols_per_row):
                    idx = row * cols_per_row + col
                    if idx < num_v_features:
                        v_values.append(v_cols[col].number_input(f"V{idx+1}", value=0.0, format="%.6f"))
        
        submit_button = st.form_submit_button("Check Transaction")
    
    if submit_button:
        # Create DataFrame with the input values
        data = pd.DataFrame([[time] + v_values + [amount]], columns=feature_names)
        
        # Make prediction
        with st.spinner("Analyzing transaction..."):
            start_time = time.time()
            prediction, probability = predict_fraud(data, model, scaler)
            end_time = time.time()
        
        # Display result
        st.markdown("### Analysis Result")
        
        # Create columns for result display
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if prediction[0] == 1:
                st.error("⚠️ **FRAUDULENT TRANSACTION DETECTED**")
                st.markdown(f"Fraud Probability: **{probability[0]:.4f}** (High Risk)")
            else:
                st.success("✅ **LEGITIMATE TRANSACTION**")
                st.markdown(f"Fraud Probability: **{probability[0]:.4f}** (Low Risk)")
            
            st.markdown(f"Processing Time: {(end_time - start_time)*1000:.2f} ms")
            
            # Risk gauge visualization
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh([0], [100], color='lightgray', height=0.3)
            ax.barh([0], [probability[0] * 100], color='red', height=0.3)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_yticks([])
            ax.set_title('Risk Level')
            st.pyplot(fig)
        
        with res_col2:
            # Show transaction details
            st.markdown("#### Transaction Details")
            st.write(f"Amount: ${amount:.2f}")
            st.write(f"Time: {time}")
            
            # Recommended action
            st.markdown("#### Recommended Action")
            if prediction[0] == 1:
                st.markdown("- Block transaction\n- Contact customer\n- Flag account for review")
            else:
                st.markdown("- Proceed with transaction\n- No action needed")

# Batch prediction page
elif page == "Batch Prediction":
    st.title("Batch Transaction Analysis")
    st.write("Upload a CSV file with multiple transactions to analyze them at once.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Check columns
        missing_cols = [col for col in feature_names if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.write("Your CSV file must contain the following columns:")
            st.write(", ".join(feature_names))
        else:
            # Process button
            if st.button("Analyze Transactions"):
                # Ensure data has the right columns in the right order
                input_data = df[feature_names].copy()
                
                # Make predictions
                with st.spinner("Analyzing transactions... This may take a moment."):
                    start_time = time.time()
                    predictions, probabilities = predict_fraud(input_data, model, scaler)
                    end_time = time.time()
                
                # Add predictions to the dataframe
                results_df = df.copy()
                results_df['Fraud_Prediction'] = predictions
                results_df['Fraud_Probability'] = probabilities
                
                # Summary statistics
                num_transactions = len(predictions)
                num_fraudulent = sum(predictions)
                fraud_rate = num_fraudulent / num_transactions * 100
                
                # Display summary
                st.subheader("Analysis Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", num_transactions)
                col2.metric("Flagged as Fraud", num_fraudulent)
                col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                
                st.write(f"Processing Time: {end_time - start_time:.2f} seconds ({(end_time - start_time) / num_transactions * 1000:.2f} ms per transaction)")
                
                # Display results
                st.subheader("Detailed Results")
                st.dataframe(results_df)
                
                # Download button for results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Distribution of fraud probabilities
                st.subheader("Fraud Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(probabilities, bins=50, kde=True, ax=ax)
                ax.set_xlabel("Fraud Probability")
                ax.set_ylabel("Number of Transactions")
                st.pyplot(fig)
                
                # High risk transactions
                high_risk = results_df[results_df['Fraud_Probability'] > 0.5].sort_values('Fraud_Probability', ascending=False)
                
                if len(high_risk) > 0:
                    st.subheader("High Risk Transactions")
                    st.dataframe(high_risk)

# Model information page
elif page == "Model Information":
    st.title("Model Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Details")
        st.write(f"**Model Type:** {model_info['model_name']}")
        st.write(f"**Model Performance (AUC-ROC):** {model_info['auc_score']:.4f}")
        st.write(f"**Last Training Date:** {model_info['training_date']}")
        
        st.subheader("Feature Importance")
        
        # Display feature importance if available
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importances
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.title('Feature Importance')
            st.pyplot(fig)
        elif hasattr(model, 'coef_'):
            # For linear models
            coefs = model.coef_[0]
            indices = np.argsort(np.abs(coefs))[::-1]
            
            # Plot feature coefficients
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.barh(range(len(indices)), coefs[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Coefficient Value')
            plt.title('Feature Coefficients')
            st.pyplot(fig)
        else:
            st.write("Feature importance not available for this model type.")
    
    with col2:
        st.subheader("How It Works")
        st.markdown("""
        This fraud detection system works by:
        
        1. **Data Collection:** Gathering transaction features (amount, time, etc.)
        
        2. **Preprocessing:** Normalizing and transforming features
        
        3. **Model Prediction:** Using machine learning to calculate fraud probability
        
        4. **Decision Making:** Flagging suspicious transactions based on threshold
        
        The model was trained on thousands of real credit card transactions with known fraud status.
        """)
        
        st.subheader("Model Performance")
        st.image("random_forest_confusion_matrix.png", caption="Confusion Matrix")
        
        st.markdown("""
        **Interpreting Results:**
        
        - **True Negatives:** Legitimate transactions correctly identified
        - **False Positives:** Legitimate transactions incorrectly flagged as fraud
        - **False Negatives:** Fraudulent transactions missed by the system
        - **True Positives:** Fraud correctly detected
        """)