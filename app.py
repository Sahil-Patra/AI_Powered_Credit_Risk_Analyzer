import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# REMOVED: from imblearn.over_sampling import SMOTE (This was causing the crash)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

# --- THE ROBUST MODEL BUILDER ---
@st.cache_resource
def build_model():
    # 1. Load Data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    names = ['status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment', 
             'installment_rate', 'personal_status_sex', 'guarantors', 'residence_since', 'property', 'age', 
             'other_installment', 'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'target']
    
    try:
        df = pd.read_csv(url, sep=' ', names=names)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, []

    # 2. Fix Target (1=Good, 2=Bad) -> (0=Good, 1=Bad)
    df['target'] = df['target'].map({1: 0, 2: 1})
    
    # 3. Preprocessing
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Train Model (Using scale_pos_weight instead of SMOTE)
    # This tells XGBoost: "Pay 3x more attention to defaults because they are rare"
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    model = xgb.XGBClassifier(
        objective='binary:logistic', 
        scale_pos_weight=ratio,  # Handles imbalance natively
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 5. Create Explainer
    explainer = shap.TreeExplainer(model)
    
    return model, explainer, scaler, feature_names

# Import numpy here just for the ratio calculation above
import numpy as np

# --- LOAD MODEL ---
with st.spinner('Building Model...'):
    model, explainer, scaler, feature_names = build_model()

if model is None:
    st.stop()

# --- DASHBOARD UI ---
st.title("🏦 AI-Powered Credit Risk Analyzer")

# Sidebar
st.sidebar.header("Applicant Data")
input_dict = {}

col1, col2 = st.sidebar.columns(2)
input_dict['duration'] = st.sidebar.slider("Loan Duration (Months)", 6, 72, 24)
input_dict['credit_amount'] = st.sidebar.number_input("Credit Amount", 500, 20000, 4000)
input_dict['age'] = st.sidebar.slider("Age", 18, 75, 30)
input_dict['installment_rate'] = st.sidebar.slider("Installment Rate (% of Income)", 1, 4, 3)

# Fill rest with defaults
for feature in feature_names:
    if feature not in input_dict:
        input_dict[feature] = 0.0

if st.button("Analyze Risk", type="primary"):
    # Process Input
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if prediction == 1:
            st.error(f"❌ **LOAN REJECTED**")
            st.write(f"Risk Probability: **{prob[1]*100:.1f}%**")
        else:
            st.success(f"✅ **LOAN APPROVED**")
            st.write(f"Approval Confidence: **{prob[0]*100:.1f}%**")

    # SHAP Plot
    st.subheader("🔍 Decision Factors")
    shap_values = explainer.shap_values(input_scaled)
    shap_explanation = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_scaled[0], feature_names=feature_names)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap_explanation, show=False)
    st.pyplot(fig)
