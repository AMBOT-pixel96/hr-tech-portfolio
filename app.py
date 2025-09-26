import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import os
from datetime import datetime

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
    # PDF Export with Executive Formatting
    # ================================
    class PDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font("Arial", "B", 12)
                self.cell(0, 10, "HR Attrition Prediction Report", ln=True, align="C")
                self.ln(5)

        def footer(self):
            self.set_y(-20)
            self.set_font("Arial", "I", 8)
            self.set_text_color(100, 100, 100)
            text = (
                "Prepared with <3 by Amlan Mishra - HR Tech, People Analytics & Compensation "
                "Management Specialist at KPMG Assurance and Consulting LLC., India"
            )
            safe_text = text.encode("latin-1", "replace").decode("latin-1")
            self.multi_cell(0, 10, safe_text, align="C")
            self.set_text_color(30, 100, 200)
            self.set_font("Arial", "U", 8)
            self.cell(
                0, 10, "Connect on LinkedIn",
                ln=True, align="C",
                link="https://www.linkedin.com/in/amlan-mishra-7aa70894"
            )

    if st.button("📑 Generate PDF Report"):
        pdf = PDF()

        # --- Cover Page ---
        pdf.add_page()
        pdf.set_font("Arial", "B", 20)
        pdf.cell(200, 15, txt="HR Attrition Prediction Report", ln=True, align="C")
        pdf.ln(15)

        pdf.set_font("Arial", "", 14)
        pdf.cell(200, 10, txt="Generated by Amlan Mishra’s Attrition Dashboard", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "I", 12)
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%d-%b-%Y %H:%M')}", ln=True, align="C")

        # --- Metrics Page ---
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="Key Metrics", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, txt=f"Model Used: {selected_model_name}", ln=True)
        pdf.cell(200, 10, txt=f"Total Employees: {total_employees}", ln=True)
        pdf.cell(200, 10, txt=f"At-Risk Employees: {at_risk}", ln=True)
        pdf.cell(200, 10, txt=f"Average Attrition Risk: {avg_prob:.1f}%", ln=True)

        # --- Snapshot Table ---
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="Employee Predictions Snapshot (Top 10)", ln=True)
        pdf.ln(8)

        # Table header
        pdf.set_font("Arial", "B", 10)
        col_widths = [40, 40, 25, 30, 25, 30]
        headers = ["JobRole", "Department", "Age", "Income", "Risk %", "Prediction"]
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, border=1, align="C")
        pdf.ln(8)

        # Table rows (with zebra striping)
        pdf.set_font("Arial", "", 9)
        fill = False
        for _, row in result_df.head(10).iterrows():
            pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.cell(col_widths[0], 8, str(row.get("JobRole", ""))[:18], border=1, fill=True)
            pdf.cell(col_widths[1], 8, str(row.get("Department", ""))[:18], border=1, fill=True)
            pdf.cell(col_widths[2], 8, str(row.get("Age", "")), border=1, align="C", fill=True)
            pdf.cell(col_widths[3], 8, str(row.get("MonthlyIncome", "")), border=1, align="R", fill=True)
            pdf.cell(col_widths[4], 8, str(row.get("Attrition_Prob", "")), border=1, align="R", fill=True)
            pdf.cell(col_widths[5], 8, str(row.get("Attrition_Pred", "")), border=1, align="C", fill=True)
            pdf.ln(8)
            fill = not fill

        # --- Charts ---
        charts = [
            ("Attrition Risk Levels", fig1_path),
            ("Attrition Probability Distribution", fig2_path),
            ("Department-Wise High-Risk Employees", fig3_path),
        ]

        for title, path in charts:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt=title, ln=True, align="C")
            pdf.image(path, x=15, y=30, w=180)

        # --- Export PDF ---
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
