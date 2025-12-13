import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

# Load model and explainer (cached for performance)
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/xgboost_credit_model.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, explainer, scaler, feature_names

try:
    model, explainer, scaler, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("Error: Model files not found. Please run 'train_model.py' locally first!")
    st.stop()

st.title("🏦 AI-Powered Credit Risk Analyzer")

# Create input form (Simplified for Demo)
st.sidebar.header("Applicant Data")
input_dict = {}

# Dynamically create inputs for the features
# (In a real app, you would make these pretty sliders/dropdowns specific to the data)
for feature in feature_names:
    input_dict[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

if st.button("Analyze Risk"):
    # Prepare input
    input_df = pd.DataFrame([input_dict])
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # SHAP
    shap_values = explainer.shap_values(input_scaled)
    
    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("❌ LOAN REJECTED (High Risk)")
        else:
            st.success("✅ LOAN APPROVED (Low Risk)")
            
    with col2:
        st.metric("Probability of Default", f"{prediction_proba[1]*100:.2f}%")
        
    # Explainability
    st.subheader("Why was this decision made?")
    
    # Waterfall Plot
    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_scaled[0],
        feature_names=feature_names
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_explanation, show=False)
    st.pyplot(fig)