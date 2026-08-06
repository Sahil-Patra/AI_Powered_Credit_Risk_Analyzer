# 🏦 AI-Powered Credit Risk Analyzer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![XGBoost](https://img.shields.io/badge/XGBoost-Native%20Categorical-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange)
![MLOps](https://img.shields.io/badge/MLOps-Decoupled%20Architecture-purple)

### 🚀 [Click Here to View Live Dashboard](https://aipoweredcreditriskanalyzer-rednfqu7pfy4rjyqx46ltt.streamlit.app/))

## 📌 Project Overview
**Problem:** Financial institutions face millions in losses annually from loan defaults. Traditional credit scoring models often act as "black boxes" that fail to explain *why* an applicant was rejected, leading to compliance risks, poor auditability, and poor customer experience.

**Solution:** This project is an end-to-end Machine Learning web application designed with enterprise MLOps patterns:
1. **Decoupled Architecture:** Separates model training (`train_model.py`) from real-time web inference (`app.py`), loading pre-compiled artifacts for zero-latency startup.
2. **Native Categorical XGBoost:** Predicts default risk using cost-sensitive gradient boosting without distorting feature distributions via manual scaling.
3. **Explainable AI (XAI):** Uses SHAP (Shapley Additive Explanations) waterfall visualizations to provide transparent, feature-level decision drivers for every loan application.

## 🛠️ Tech Stack
* **Frontend / UI:** Streamlit
* **Model Training & Artifacts:** XGBoost (`enable_categorical=True`), Joblib, Scikit-Learn
* **Explainability:** SHAP (TreeExplainer)
* **Data Processing:** Pandas, NumPy
* **Dataset:** German Credit Dataset (UCI Machine Learning Repository)

## 📊 Key Features & MLOps Architecture
* **Decoupled Train-Inference Pipeline:** Pre-trained models, feature names, and statistical imputation defaults are serialized to `.pkl` files, eliminating web app startup latency.
* **Instant Risk Probability Scoring:** Real-time credit default prediction with calibrated risk percentages.
* **Feature-Level Explainability:** SHAP waterfall charts showing positive and negative forces driving approval/rejection decisions.
* **Statistical Imputation Safeguards:** Fills unselected UI features with training set medians and modes (preventing silent inference corruption from magic numbers).
* **Memory-Safe Plot Rendering:** Explicit figure lifecycle management (`plt.close()`) preventing memory leaks during Streamlit UI reruns.

## 📂 Project Structure
```bash
CreditRiskProject/
├── models/                      # Serialized ML artifacts (.pkl files)
│   ├── xgboost_credit_model.pkl # Trained XGBoost classifier
│   ├── shap_explainer.pkl       # Serialized SHAP TreeExplainer
│   ├── feature_names.pkl        # Model feature schema
│   └── feature_defaults.pkl     # Medians & modes for statistical imputation
├── app.py                       # Real-time Streamlit web dashboard
├── train_model.py               # Offline training & artifact serialization pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```
⚙️ How to Run Locally
Clone the repository:
```Bash
git clone https://github.com/Sahil-Patra/AI_Powered_Credit_Risk_Analyzer.git
```
Install dependencies:
```Bash
pip install -r requirements.txt
```
Run the dashboard:
```Bash
streamlit run app.py
```

## 📈 Model Performance & Engineering Highlights

**Class Imbalance Optimization**: Utilizes native `scale_pos_weight` tuning based on dataset default ratios to prioritize recall on high-risk borrowers.
**Auditability**: Validated top predictors (Duration, Credit Amount, Savings, Employment Status) align with Basel III banking risk standards.
