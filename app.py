import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")
st.title("🚀 HR Attrition Prediction Dashboard")

# ================================
# Load Models
# ================================
MODEL_PATH = "models_joblib/"

models = {
    "Logistic Regression": joblib.load(MODEL_PATH + "logistic_attrition_model.joblib"),
    "Random Forest": joblib.load(MODEL_PATH + "random_forest_attrition_model.joblib"),
    "XGBoost": joblib.load(MODEL_PATH + "xgboost_attrition_model.joblib"),
}

scaler = joblib.load(MODEL_PATH + "scaler.joblib")
trained_columns = joblib.load(MODEL_PATH + "trained_columns.joblib")

# ================================
# Sidebar - Model Selection
# ================================
st.sidebar.header("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
selected_model = models[selected_model_name]

st.sidebar.success(f"✅ Using {selected_model_name} for predictions")

# ================================
# Upload Section
# ================================
st.subheader("📂 Upload Employee Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV in IBM HR Analytics format", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Align columns
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[trained_columns]

    X_scaled = scaler.transform(df)

    y_pred = selected_model.predict(X_scaled)
    y_prob = selected_model.predict_proba(X_scaled)[:, 1]

    df["Attrition_Pred"] = y_pred
    df["Attrition_Prob"] = y_prob

    # ================================
    # Metrics
    # ================================
    total_employees = len(df)
    at_risk = int((df["Attrition_Pred"] == 1).sum())
    avg_prob = df["Attrition_Prob"].mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Total Employees", total_employees)
    c2.metric("⚠️ At-Risk Employees", at_risk)
    c3.metric("📈 Avg Attrition Risk", f"{avg_prob:.1f}%")

    # Results table
    display_cols = ["Age", "Department", "JobRole", "MonthlyIncome", "Attrition_Prob", "Attrition_Pred"]
    result_df = df[display_cols].copy()
    result_df["Attrition_Prob"] = (result_df["Attrition_Prob"] * 100).round(1)
    result_df["Attrition_Pred"] = result_df["Attrition_Pred"].map({1: "🚨 At Risk", 0: "✅ Safe"})

    st.write("### 📋 Employee Risk Predictions")
    st.dataframe(result_df, use_container_width=True)

    # Visuals
    st.write("### 📊 Attrition Risk Insights")
    risk_levels = pd.cut(df["Attrition_Prob"], bins=[0, 0.33, 0.66, 1], labels=["Low", "Medium", "High"])
    risk_counts = risk_levels.value_counts()

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax = plt.subplots()
        sns.barplot(x=risk_counts.index, y=risk_counts.values,
                    palette=["green", "orange", "red"], ax=ax)
        ax.set_ylabel("Employees")
        st.pyplot(fig1)

    with col2:
        fig2, ax = plt.subplots()
        sns.histplot(df["Attrition_Prob"], bins=20, kde=True, color="blue", ax=ax)
        ax.set_xlabel("Attrition Probability")
        st.pyplot(fig2)

    # ================================
    # CSV Export
    # ================================
    st.download_button(
        label="💾 Download Results (CSV)",
        data=result_df.to_csv(index=False),
        file_name=f"attrition_predictions_{selected_model_name.replace(' ','_')}.csv",
        mime="text/csv",
    )

    # ================================
    # PDF Export
    # ================================
    if st.button("📑 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)

        pdf.cell(200, 10, txt="HR Attrition Prediction Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Model Used: {selected_model_name}", ln=True)
        pdf.cell(200, 10, txt=f"Total Employees: {total_employees}", ln=True)
        pdf.cell(200, 10, txt=f"At-Risk Employees: {at_risk}", ln=True)
        pdf.cell(200, 10, txt=f"Average Attrition Risk: {avg_prob:.1f}%", ln=True)
        pdf.ln(10)

        pdf.multi_cell(0, 10, txt="Top Predictions Snapshot:")
        for i, row in result_df.head(10).iterrows():
            pdf.cell(0, 10, txt=f"{row['JobRole']} ({row['Department']}) - Risk {row['Attrition_Prob']}% ({row['Attrition_Pred']})", ln=True)

        # Export as downloadable PDF
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_bytes = pdf_buffer.getvalue()

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"attrition_report_{selected_model_name.replace(' ','_')}.pdf",
            mime="application/pdf",
        )

else:
    st.info("⬆️ Upload a CSV file to start predictions.")