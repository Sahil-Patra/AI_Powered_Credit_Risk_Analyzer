import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_and_export():
    print("📥 Loading German Credit Dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    names = [
        'status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings', 
        'employment', 'installment_rate', 'personal_status_sex', 'guarantors', 
        'residence_since', 'property', 'age', 'other_installment', 'housing', 
        'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'target'
    ]
    
    df = pd.read_csv(url, sep=' ', names=names)
    df['target'] = df['target'].map({1: 0, 2: 1}) # 0 = Good, 1 = Default Risk

    X = df.drop('target', axis=1)
    y = df['target']

    # Convert categorical columns to category dtype for native XGBoost support
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate native imbalance weight (No SMOTE needed for XGBoost)
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

    print("⚡ Training Native XGBoost Model with Categorical Support...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        enable_categorical=True, # Built-in categorical handling (No scaling/manual coding needed!)
        scale_pos_weight=ratio,
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    print("\n📊 Model Evaluation:")
    print(classification_report(y_test, preds))

    # Calculate Feature Defaults (Medians for numbers, Modes for categories) for Inference Imputation
    defaults = {}
    for col in X.columns:
        if X[col].dtype.name == 'category':
            defaults[col] = X[col].mode()[0]
        else:
            defaults[col] = X[col].median()

    print("🔍 Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    # Save Production Artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_credit_model.pkl')
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
    joblib.dump(defaults, 'models/feature_defaults.pkl')
    
    print("✅ All artifacts successfully exported to 'models/' directory!")

if __name__ == "__main__":
    train_and_export()