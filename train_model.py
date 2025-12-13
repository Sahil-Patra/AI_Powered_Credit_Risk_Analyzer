import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# 1. LOAD DATA (Using the URL directly so you don't need to download files)
# We are using the "Give Me Some Credit" dataset logic but adapting for the German Credit context in your guide
# or simply using the German dataset as suggested in the guide:
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
names = ['status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment', 
         'installment_rate', 'personal_status_sex', 'guarantors', 'residence_since', 'property', 'age', 
         'other_installment', 'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'target']

df = pd.read_csv(url, sep=' ', names=names)

# Fix Target: In this dataset 1=Good, 2=Bad. We want 0=Good, 1=Bad (Default)
df['target'] = df['target'].map({1: 0, 2: 1})

print("Data loaded. Rows:", len(df))

# 2. PREPROCESSING (Phase 1 from guide)
# Simple encoding for this demo
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle Imbalance (SMOTE)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Save the feature names for later use in the app
feature_names = X.columns.tolist()

# 3. TRAIN MODEL (Phase 2 from guide)
print("Training XGBoost Model...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1
)
model.fit(X_train_scaled, y_train_balanced)

print("Model Trained.")

# 4. EXPLAINABILITY (Phase 3 from guide)
print("Generating SHAP Explainer...")
explainer = shap.TreeExplainer(model)

# 5. SAVE ARTIFACTS (Crucial for app.py)
# Create a 'models' folder first if it doesn't exist
import os
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/xgboost_credit_model.pkl')
joblib.dump(explainer, 'models/shap_explainer.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl') # Saving feature names too

print("SUCCESS! All .pkl files saved in 'models/' folder.")