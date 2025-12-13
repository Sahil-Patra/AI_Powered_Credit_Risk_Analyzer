import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- PAGE CONFIG ---
st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

# --- THE NUCLEAR OPTION: TRAIN ON THE FLY ---
# We use @st.cache_resource so this ONLY runs once when the app starts.
# It won't re-train every time you click a button.
@st.cache_resource
def build_model():
    # 1. Load Data directly from the web
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
    # Encode text columns to numbers
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle Imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    
    # 4. Train Model
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train_scaled, y_train_balanced)
    
    # 5. Create Explainer
    explainer = shap.TreeExplainer(model)
    
    return model, explainer, scaler, feature_names

# --- LOAD THE MODEL ---
# This line triggers the training (only takes 10 seconds the first time)
with st.spinner('Waking up the AI... (Training Model on Cloud)'):
    model, explainer, scaler, feature_names = build_model()

if model is None:
    st.stop()

# --- THE DASHBOARD ---
st.title("🏦 AI-Powered Credit Risk Analyzer")
st.markdown("This dashboard uses **XGBoost** and **SHAP** to predict credit risk and explain the decision.")

# Sidebar Inputs
st.sidebar.header("Applicant Data")
input_dict = {}

# Create inputs for key features
# We hardcode specific ones for better UX, defaulting others to median/mode
col1, col2 = st.sidebar.columns(2)

# Key Drivers
input_dict['duration'] = st.sidebar.slider("Loan Duration (Months)", 6, 72, 24)
input_dict['credit_amount'] = st.sidebar.number_input("Credit Amount", 500, 20000, 4000)
input_dict['age'] = st.sidebar.slider("Age", 18, 75, 30)
input_dict['installment_rate'] = st.sidebar.slider("Installment Rate (% of Income)", 1, 4, 3)

# Default the rest (Hidden from simplified UI for speed)
for feature in feature_names:
    if feature not in input_dict:
        input_dict[feature] = 0.0 # Default value for other technical columns

# Prediction Logic
if st.button("Analyze Risk", type="primary"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Align columns to match training data
    input_df = input_df[feature_names]
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # --- RESULTS ---
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.error("❌ **LOAN REJECTED** (High Risk)")
            st.write(f"Confidence: **{prediction_proba[1]*100:.1f}%**")
        else:
            st.success("✅ **LOAN APPROVED** (Low Risk)")
            st.write(f"Confidence: **{prediction_proba[0]*100:.1f}%**")

    # --- EXPLAINABILITY ---
    st.subheader("🔍 Why was this decision made?")
    
    # Calculate SHAP
    shap_values = explainer.shap_values(input_scaled)
    
    # Waterfall Plot
    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_scaled[0],
        feature_names=feature_names
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    shap.waterfall_plot(shap_explanation, show=False)
    st.pyplot(fig)
