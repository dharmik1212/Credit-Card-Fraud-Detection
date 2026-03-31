
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model, scaler, and selected features
@st.cache_resource # Cache the model loading for better performance
def load_resources():
    best_rf_model = joblib.load('best_rf_model.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    selected_features = joblib.load('selected_features.pkl')
    return best_rf_model, scaler, selected_features

best_rf_model, scaler, selected_features = load_resources()

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

st.markdown("--- ")
st.header("Enter Transaction Details")

# Input fields for features
with st.expander("Input Transaction Features"): # Use an expander for better UI organization
    # Generate input fields dynamically for all selected features
    input_values = {}
    num_cols = 3 # Number of columns for layout
    cols = st.columns(num_cols)
    col_idx = 0

    for feature in selected_features:
        with cols[col_idx]:
            if feature == 'Time' or feature == 'Amount':
                input_values[feature] = st.number_input(f'{feature}', value=100.0, format="%.4f")
            else:
                # For V-features, provide a reasonable default or random value
                input_values[feature] = st.number_input(f'{feature}', value=0.0, format="%.4f")
        col_idx = (col_idx + 1) % num_cols


# --- Prediction Logic ---
if st.button("Predict Fraud"): # Button to trigger prediction
    # Convert input to DataFrame
    raw_input_df = pd.DataFrame([input_values])

    # Preprocess 'Time' and 'Amount' using the loaded scaler
    try:
        df_to_scale = raw_input_df[['Time', 'Amount']].copy()
        df_scaled = scaler.transform(df_to_scale)
        raw_input_df['Time'] = df_scaled[:, 0]
        raw_input_df['Amount'] = df_scaled[:, 1]
    except Exception as e:
        st.error(f"Error during scaling: {e}. Make sure the scaler is correctly loaded and features are present.")
        st.stop()

    # Select and order features for the model
    model_input_df = raw_input_df[selected_features]

    # Make prediction
    prediction = best_rf_model.predict(model_input_df)[0]
    prediction_proba = best_rf_model.predict_proba(model_input_df)[0][1]

    st.markdown("--- ")
    st.header("Prediction Result:")
    if prediction == 1:
        st.error(f"### 🚨 Fraudulent Transaction Detected! (Probability: {prediction_proba:.4f})")
        st.balloons()
    else:
        st.success(f"### ✅ Legitimate Transaction. (Probability: {prediction_proba:.4f})")

    st.write(f"Raw input received: {input_values}")
    st.write(f"Processed input for model: {model_input_df.to_dict('records')[0]}")
