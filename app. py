import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------
# Load Models & Scaler
# -----------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    logistic = joblib.load("models/logistic_attrition_model.pkl")
    rf = joblib.load("models/random_forest_tuned.pkl")
    xgb = joblib.load("models/xgboost_attrition_model.pkl")
    return scaler, logistic, rf, xgb

scaler, logistic, rf, xgb = load_models()

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Attrition Dashboard", layout="wide")
st.title("📊 Employee Attrition Prediction Dashboard")

# -----------------------
# Sidebar: Model Selector
# -----------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])

# -----------------------
# File Upload
# -----------------------
st.header("📂 Upload Employee Dataset")
uploaded_file = st.file_uploader("Upload a CSV (IBM HR Analytics format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 👀 Preview of Data", df.head())

    # Preprocess
    X = df.drop(columns=["Attrition"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    # Align with training features
    # (fill missing columns if mismatch)
    trained_cols = joblib.load("models/trained_columns.pkl")  # Save this during training
    for col in trained_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[trained_cols]

    # -----------------------
    # Model Predictions
    # -----------------------
    if model_choice == "Logistic Regression":
        X_scaled = scaler.transform(X)
        preds = logistic.predict(X_scaled)
        probs = logistic.predict_proba(X_scaled)[:,1]
    elif model_choice == "Random Forest":
        preds = rf.predict(X)
        probs = rf.predict_proba(X)[:,1]
    else:
        preds = xgb.predict(X)
        probs = xgb.predict_proba(X)[:,1]

    df_results = df.copy()
    df_results["Attrition_Pred"] = preds
    df_results["Attrition_Prob"] = probs

    st.write("### 📝 Prediction Results", df_results.head())

    # -----------------------
    # SHAP Explanations (Optional)
    # -----------------------
    with st.expander("🔍 SHAP Model Explainability"):
        explainer = shap.Explainer(xgb, X)
        shap_values = explainer(X)

        st.write("#### Global Feature Importance")
        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig1)

        st.write("#### Local Explanation for First Employee")
        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig2)
