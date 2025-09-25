import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import os

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

st.sidebar.success(f"Using {selected_model_name} for predictions")

# ================================
# Upload Section
# ================================
st.subheader("📂 Upload Employee Dataset")

uploaded_file = st.file_uploader("Upload CSV (HRIS-style export)", type="csv")

# Download Sample CSV (blank headers)
sample_headers = ["Age", "Department", "JobRole", "MonthlyIncome"]
sample_csv = pd.DataFrame(columns=sample_headers).to_csv(index=False)

st.download_button(
    label="📥 Download Sample CSV Template",
    data=sample_csv,
    file_name="sample_hris_upload.csv",
    mime="text/csv",
)

if uploaded_file:
    # Keep original for HR-friendly display
    raw_df = pd.read_csv(uploaded_file)

    # Align for model
    df = raw_df.copy()
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[trained_columns]

    X_scaled = scaler.transform(df)

    y_pred = selected_model.predict(X_scaled)
    y_prob = selected_model.predict_proba(X_scaled)[:, 1]

    raw_df["Attrition_Pred"] = y_pred
    raw_df["Attrition_Prob"] = y_prob

    # ================================
    # Metrics
    # ================================
    total_employees = len(raw_df)
    at_risk = int((raw_df["Attrition_Pred"] == 1).sum())
    avg_prob = raw_df["Attrition_Prob"].mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Employees", total_employees)
    c2.metric("At-Risk Employees", at_risk)
    c3.metric("Average Attrition Risk", f"{avg_prob:.1f}%")

    # ================================
    # Results Table
    # ================================
    display_cols = ["Age", "Department", "JobRole", "MonthlyIncome", "Attrition_Prob", "Attrition_Pred"]
    result_df = raw_df[display_cols].copy()
    result_df["Attrition_Prob"] = (result_df["Attrition_Prob"] * 100).round(1)
    result_df["Attrition_Pred"] = result_df["Attrition_Pred"].map({1: "At Risk", 0: "Safe"})

    st.write("### 📋 Employee Risk Predictions")
    st.dataframe(result_df, use_container_width=True)

    # ================================
    # Visuals
    # ================================
    st.write("### 📊 Attrition Risk Insights")

    # Risk distribution
    risk_levels = pd.cut(raw_df["Attrition_Prob"], bins=[0, 0.33, 0.66, 1], labels=["Low", "Medium", "High"])
    risk_counts = risk_levels.value_counts()

    fig1, ax1 = plt.subplots()
    sns.barplot(x=risk_counts.index, y=risk_counts.values,
                palette=["green", "orange", "red"], ax=ax1)
    ax1.set_ylabel("Employees")
    st.pyplot(fig1)

    # Probability distribution
    fig2, ax2 = plt.subplots()
    sns.histplot(raw_df["Attrition_Prob"], bins=20, kde=True, color="blue", ax=ax2)
    ax2.set_xlabel("Attrition Probability")
    st.pyplot(fig2)

    # Department-wise high risk
    high_risk_df = raw_df[raw_df["Attrition_Pred"] == "At Risk"]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    if not high_risk_df.empty and "Department" in high_risk_df.columns:
        sns.countplot(x="Department", data=high_risk_df, palette="Reds", ax=ax3)
        plt.xticks(rotation=30)
        ax3.set_ylabel("High-Risk Employees")
    else:
        ax3.text(0.5, 0.5, "No high-risk employees detected", ha="center", va="center")
    st.pyplot(fig3)

    # Save charts for PDF
    os.makedirs("temp_charts", exist_ok=True)
    fig1_path = "temp_charts/risk_distribution.png"
    fig2_path = "temp_charts/probability_distribution.png"
    fig3_path = "temp_charts/department_high_risk.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")

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
    # PDF Export with Footer Branding
    # ================================
    class PDF(FPDF):
        def footer(self):
            self.set_y(-20)
            self.set_font("Arial", "I", 8)
            self.set_text_color(100, 100, 100)

            # Branding line (ASCII only)
            text = (
                "Prepared with <3 by Amlan Mishra - HR Tech, People Analytics & Compensation "
                "Management Specialist at KPMG Assurance and Consulting LLC., India"
            )
            safe_text = text.encode("latin-1", "replace").decode("latin-1")
            self.multi_cell(0, 10, safe_text, align="C")

            # LinkedIn link
            self.set_text_color(30, 100, 200)  # link blue
            self.set_font("Arial", "U", 8)
            self.cell(
                0, 10, "Connect on LinkedIn",
                ln=True, align="C",
                link="https://www.linkedin.com/in/amlan-mishra-7aa70894"
            )

    if st.button("📑 Generate PDF Report"):
        pdf = PDF()
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

        pdf.multi_cell(0, 10, txt="Top Predictions Snapshot (10 rows):")
        for i, row in result_df.head(10).iterrows():
            pdf.cell(
                0, 10,
                txt=f"{row.get('JobRole','N/A')} ({row.get('Department','N/A')}) - Risk {row['Attrition_Prob']}% ({row['Attrition_Pred']})",
                ln=True
            )

        # Insert charts
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Charts", ln=True, align="C")
        pdf.image(fig1_path, x=10, y=30, w=180)
        pdf.ln(100)
        pdf.image(fig2_path, x=10, y=140, w=180)
        pdf.add_page()
        pdf.cell(200, 10, txt="Department-Wise High-Risk Employees", ln=True, align="C")
        pdf.image(fig3_path, x=10, y=40, w=180)

        # Export PDF
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
    st.info("⬆️ Upload a CSV or use the Sample Template to start predictions.")