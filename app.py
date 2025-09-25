import streamlit as st
import pandas as pd
import pickle

# 🚀 Title
st.title("🚀 HR Attrition Prediction Dashboard")

# 📂 Load models & scaler with pickle
with open("models/logistic_attrition_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("models/random_forest_tuned.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/xgboost_attrition_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/trained_columns.pkl", "rb") as f:
    trained_columns = pickle.load(f)

# 📤 File uploader
uploaded_file = st.file_uploader("Upload HR Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Ensure columns match training
    df = pd.get_dummies(df)
    df = df.reindex(columns=trained_columns, fill_value=0)

    # 🔄 Scale numeric features
    X_scaled = scaler.transform(df)

    # 🧠 Predictions
    logistic_pred = logistic_model.predict(X_scaled)
    rf_pred = rf_model.predict(X_scaled)
    xgb_pred = xgb_model.predict(X_scaled)

    # 📊 Show results
    st.subheader("Predictions")
    df["Logistic_Pred"] = logistic_pred
    df["RandomForest_Pred"] = rf_pred
    df["XGBoost_Pred"] = xgb_pred

    st.dataframe(df.head())
else:
    st.info("👆 Upload a CSV file to start predictions.")