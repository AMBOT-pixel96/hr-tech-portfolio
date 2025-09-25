import streamlit as st

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Attrition Dashboard", layout="wide")
st.title("🚀 HR Attrition Prediction Dashboard")

# -----------------------
# Import libraries safely
# -----------------------
try:
    import pandas as pd
    import joblib
    import shap
    import matplotlib.pyplot as plt
    st.info("📦 Libraries imported successfully")
except Exception as e:
    st.error(f"❌ Import failed: {e}")
    st.stop()

# -----------------------
# Load Models
# -----------------------
try:
    scaler = joblib.load("models/scaler.pkl")
    logistic = joblib.load("models/logistic_attrition_model.pkl")
    rf = joblib.load("models/random_forest_tuned.pkl")
    xgb = joblib.load("models/xgboost_attrition_model.pkl")
    trained_cols = joblib.load("models/trained_columns.pkl")
    st.success("✅ Models and scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# -----------------------
# Sidebar: Model Selector
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

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.info(f"📂 Uploaded file with {df.shape[0]} rows, {df.shape[1]} columns")
        st.write("### 👀 Preview of Data", df.head())
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()

    # -----------------------
    # Preprocess
    # -----------------------
    try:
        X = df.drop(columns=["Attrition"], errors="ignore")
        X = pd.get_dummies(X, drop_first=True)

        # Align with training features
        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[trained_cols]

        st.success("✅ Preprocessing complete. Features aligned.")
    except Exception as e:
        st.error(f"❌ Preprocessing failed: {e}")
        st.stop()

    # -----------------------
    # Model Predictions
    # -----------------------
    try:
        preds, probs = None, None
        if model_choice == "Logistic Regression":
            X_scaled = scaler.transform(X)
            preds = logistic.predict(X_scaled)
            probs = logistic.predict_proba(X_scaled)[:, 1]
        elif model_choice == "Random Forest":
            preds = rf.predict(X)
            probs = rf.predict_proba(X)[:, 1]
        else:
            preds = xgb.predict(X)
            probs = xgb.predict_proba(X)[:, 1]

        df_results = df.copy()
        df_results["Attrition_Pred"] = preds
        df_results["Attrition_Prob"] = probs

        st.success("✅ Predictions generated")
        st.write("### 📝 Prediction Results", df_results.head())
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # -----------------------
    # SHAP Explanations
    # -----------------------
    try:
        with st.expander("🔍 SHAP Model Explainability"):
            st.info("⚡ Running SHAP explainer (this may take a moment)...")
            explainer = shap.Explainer(xgb, X.sample(100, random_state=42))
            shap_values = explainer(X.sample(100, random_state=42))

            st.write("#### Global Feature Importance")
            fig1, ax1 = plt.subplots()
            shap.summary_plot(shap_values, X.sample(100, random_state=42), show=False)
            st.pyplot(fig1)

            st.write("#### Local Explanation (First Employee in Sample)")
            fig2, ax2 = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig2)

            st.success("✅ SHAP visualizations generated")
    except Exception as e:
        st.warning(f"⚠️ SHAP explainability skipped due to error: {e}")