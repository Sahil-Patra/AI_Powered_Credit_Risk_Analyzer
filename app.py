import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

# --- LOAD PRE-TRAINED ARTIFACTS ---
@st.cache_resource
def load_ml_artifacts():
    """Loads serialized pipeline artifacts instantly from local storage."""
    try:
        model = joblib.load('models/xgboost_credit_model.pkl')
        explainer = joblib.load('models/shap_explainer.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        defaults = joblib.load('models/feature_defaults.pkl')
        return model, explainer, feature_names, defaults
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}. Please run 'train_model.py' first.")
        return None, None, None, None

model, explainer, feature_names, feature_defaults = load_ml_artifacts()

if model is None:
    st.stop()

st.title("🏦 AI-Powered Credit Risk Analyzer")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Applicant Details")
user_inputs = {}

# User Editable Features
user_inputs['duration'] = st.sidebar.slider("Loan Duration (Months)", 6, 72, 24)
user_inputs['credit_amount'] = st.sidebar.number_input("Credit Amount (DM)", 500, 20000, 4000)
user_inputs['age'] = st.sidebar.slider("Age", 18, 75, 30)
user_inputs['installment_rate'] = st.sidebar.slider("Installment Rate (% of Income)", 1, 4, 3)

# Categorical Dropdowns mapped to exact dataset category string formats
user_inputs['employment'] = st.sidebar.selectbox(
    "Employment Duration",
    ["A71", "A72", "A73", "A74", "A75"],
    format_func=lambda x: {"A71": "Unemployed", "A72": "< 1 Year", "A73": "1-4 Years", "A74": "4-7 Years", "A75": ">= 7 Years"}[x]
)

user_inputs['savings'] = st.sidebar.selectbox(
    "Savings Balance",
    ["A61", "A62", "A63", "A64", "A65"],
    format_func=lambda x: {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "Unknown/No Account"}[x]
)

user_inputs['property'] = st.sidebar.selectbox(
    "Property Owned",
    ["A121", "A122", "A123", "A124"],
    format_func=lambda x: {"A121": "Real Estate", "A122": "Building Society", "A123": "Car/Other", "A124": "None"}[x]
)

# Populate remaining unselected features with true Statistical Defaults (Medians/Modes)
final_input = {}
for feat in feature_names:
    if feat in user_inputs:
        final_input[feat] = user_inputs[feat]
    else:
        final_input[feat] = feature_defaults[feat]

# --- INFERENCE & SHAP EXPLAINABILITY ---
if st.button("Analyze Credit Risk", type="primary"):
    # Construct DataFrame with exact categorical dtypes expected by XGBoost
    input_df = pd.DataFrame([final_input])[feature_names]
    
    # Ensure categorical columns are properly typed for native XGBoost inference
    for col in input_df.columns:
        if isinstance(feature_defaults[col], str):
            input_df[col] = input_df[col].astype('category')

    # Model Inference
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    risk_prob = probabilities[1]

    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Risk Assessment Result")
        if prediction == 1:
            st.error(f"❌ **HIGH RISK - LOAN REJECTED**")
            st.metric("Default Risk Probability", f"{risk_prob * 100:.1f}%")
        else:
            st.success(f"✅ **LOW RISK - LOAN APPROVED**")
            st.metric("Approval Confidence", f"{(1 - risk_prob) * 100:.1f}%")

    with col2:
        st.subheader("🔍 SHAP Explanation (Decision Factors)")
        # Calculate SHAP values on raw inputs
        shap_values = explainer(input_df)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig) # Prevent Matplotlib memory leaks