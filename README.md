# 🏦 AI-Powered Credit Risk Analyzer

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange)

### 🚀 [Click Here to View Live Dashboard](https://share.streamlit.io/YOUR_USERNAME/REPO_NAME)

## 📌 Project Overview
**Problem:** Banks lose millions annually due to loan defaults. Traditional credit scoring models are often "black boxes" that fail to explain *why* an applicant was rejected, leading to regulatory issues and poor customer experience.

**Solution:** This project is an end-to-end Machine Learning web application that:
1.  **Predicts Loan Default Risk** using a robust XGBoost classifier.
2.  **Explains Decisions** using SHAP (Shapley Additive Explanations) values, providing transparency on exactly which factors (e.g., Age, Debt-to-Income) contributed to the rejection.
3.  **Reduces Risk:** Prioritizes identifying high-risk borrowers to minimize financial loss.

## 🛠️ Tech Stack
*   **Frontend:** Streamlit (Web Dashboard)
*   **Backend:** Python, Pandas
*   **Machine Learning:** XGBoost (Gradient Boosting), Scikit-Learn
*   **Explainability:** SHAP (Game Theoretic approach to feature importance)
*   **Data:** German Credit Dataset (UCI Machine Learning Repository)

## 📊 Key Features
*   **Real-time Risk Assessment:** Instant Yes/No prediction with probability score.
*   **Interactive "Why?" Analysis:** Waterfall charts showing exactly why a user was rejected.
*   **User-Friendly Interface:** Simple sidebar inputs for non-technical users (Loan Officers).

## 📂 Project Structure
```bash
CreditRiskProject/
├── models/                  # Trained models (.pkl files)
├── app.py                   # Streamlit Dashboard application
├── train_model.py           # ML Pipeline (Data -> Preprocessing -> Training)
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```
⚙️ How to Run Locally
Clone the repository:
```Bash
git clone https://github.com/YOUR_USERNAME/CreditRiskProject.git
```
Install dependencies:
```Bash
pip install -r requirements.txt
```
Run the dashboard:
```Bash
streamlit run app.py
```

## 📈 Model Performance

Recall (Sensitivity): Optimized to catch potential defaulters.
Explainability: Validated top predictors (Duration, Credit Amount, Checking Status) match banking domain knowledge.
