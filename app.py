import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------
# Config & Paths
# -----------------------
st.set_page_config(page_title="Attrition Dashboard", layout="wide")
st.title("🚀 HR Attrition Prediction Dashboard")

MODEL_DIR = Path(__file__).parent / "models_joblib"

# -----------------------
# Load Models (joblib)
# -----------------------
def load_model(name):
    path = MODEL_DIR / name
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Could not load {name}: {e}")
        return None

logistic_model = load_model("logistic_attrition_model.joblib")
rf_model       = load_model("random_forest_attrition_model.joblib")
xgb_model      = load_model("xgboost_attrition_model.joblib")
scaler         = load_model("scaler.joblib")
trained_cols   = load_model("trained_columns.joblib")

if not all([logistic_model, rf_model, xgb_model, scaler, trained_cols]):
    st.stop()

st.success("✅ All models loaded successfully from /models_joblib/")

# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"]
)

# -----------------------
# File Upload
# -----------------------
st.header("📂 Upload Employee Dataset")
uploaded_file = st.file_uploader("Upload a CSV (IBM HR Analytics format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 👀 Data Preview", df.head())

    # Preprocess
    X = df.drop(columns=["Attrition"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    # Align features with training
    for col in trained_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[trained_cols]

    # -----------------------
    # Prediction
    # -----------------------
    preds, probs = None, None
    if model_choice == "Logistic Regression":
        X_scaled = scaler.transform(X)
        preds = logistic_model.predict(X_scaled)
        probs = logistic_model.predict_proba(X_scaled)[:, 1]
    elif model_choice == "Random Forest":
        preds = rf_model.predict(X)
        probs = rf_model.predict_proba(X)[:, 1]
    else:
        preds = xgb_model.predict(X)
        probs = xgb_model.predict_proba(X)[:, 1]

    # Results
    df_results = df.copy()
    df_results["Attrition_Pred"] = preds
    df_results["Attrition_Prob"] = probs

    st.write("### 📝 Prediction Results", df_results.head())
else:
    st.info("👆 Upload a CSV file to start predictions.")
