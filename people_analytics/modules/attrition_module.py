# ============================================
# modules/attrition_module.py — v1.0
# Attrition Analytics Module
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_helper import render_pdf_download_button

def run_attrition_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#991B1B,#DC2626);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">📉 Attrition Analytics</h2>
        <p style="font-size:14px;margin-top:6px;">
        Analyze workforce attrition trends, identify risk areas, and visualize retention insights by department, level, and demographics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================
    # Step 1: Download Template
    # ==========================
    st.subheader("📄 Step 1 — Download Attrition Data Template")

    sample_data = pd.DataFrame({
        "EmployeeID": ["E1001", "E1002"],
        "Department": ["Finance", "IT"],
        "JobLevel": ["Analyst", "Manager"],
        "Gender": ["Male", "Female"],
        "TenureYears": [2.5, 4.0],
        "CTC": [600000, 950000],
        "AttritionFlag": [1, 0]  # 1 = Left, 0 = Active
    })
    render_download_template("Attrition Data Template", sample_data, "Attrition_Template.csv")

    # ==========================
    # Step 2: Upload Data
    # ==========================
    st.subheader("📤 Step 2 — Upload Attrition Data")
    uploaded = st.file_uploader("Upload filled attrition dataset (CSV)", type=["csv"])

    if not uploaded:
        st.info("Please upload the completed attrition file to continue.")
        return

    try:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # ==========================
    # Step 3: Validation
    # ==========================
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender", "TenureYears", "CTC", "AttritionFlag"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Cleanup & numeric conversions
    df["CTC"] = pd.to_numeric(df["CTC"], errors="coerce")
    df["TenureYears"] = pd.to_numeric(df["TenureYears"], errors="coerce")
    df["AttritionFlag"] = pd.to_numeric(df["AttritionFlag"], errors="coerce").astype(int)
    df = df.dropna(subset=["CTC", "AttritionFlag"])

    # ==========================
    # Step 4: Metrics
    # ==========================
    st.subheader("📊 Attrition Insights Dashboard")

    total_employees = len(df)
    total_attrition = df["AttritionFlag"].sum()
    attrition_rate = (total_attrition / total_employees) * 100
    avg_tenure = df["TenureYears"].mean()
    avg_ctc = df["CTC"].mean() / 1e5

    st.metric("Overall Attrition Rate", f"{attrition_rate:.2f}%")
    st.metric("Average Tenure (Years)", f"{avg_tenure:.1f}")
    st.metric("Average CTC (₹ Lakhs)", f"{avg_ctc:.2f}")

    # --- Attrition by Department
    st.markdown("#### 🏢 Attrition Rate by Department")
    dept_summary = df.groupby("Department", observed=True)["AttritionFlag"].mean().mul(100).round(2).reset_index()
    fig_dept = px.bar(dept_summary, x="Department", y="AttritionFlag", color="Department",
                      text="AttritionFlag", title="Attrition Rate by Department")
    fig_dept.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_dept, use_container_width=True)

    # --- Attrition by Job Level
    st.markdown("#### 💼 Attrition Rate by Job Level")
    level_summary = df.groupby("JobLevel", observed=True)["AttritionFlag"].mean().mul(100).round(2).reset_index()
    fig_level = px.bar(level_summary, x="JobLevel", y="AttritionFlag", color="JobLevel",
                       text="AttritionFlag", title="Attrition Rate by Job Level")
    fig_level.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_level, use_container_width=True)

    # --- Gender-wise Attrition
    st.markdown("#### ⚖️ Gender-wise Attrition")
    gender_summary = df.groupby("Gender", observed=True)["AttritionFlag"].mean().mul(100).round(2).reset_index()
    fig_gender = px.pie(gender_summary, values="AttritionFlag", names="Gender", title="Attrition Distribution by Gender")
    st.plotly_chart(fig_gender, use_container_width=True)

    # --- Tenure Distribution
    st.markdown("#### ⏳ Tenure Distribution (in Years)")
    fig_tenure = px.histogram(df, x="TenureYears", nbins=10, color_discrete_sequence=["#2563EB"],
                              title="Tenure Distribution")
    st.plotly_chart(fig_tenure, use_container_width=True)

    # --- CTC vs Attrition
    st.markdown("#### 💰 CTC vs Attrition")
    ctc_summary = df.groupby("AttritionFlag", observed=True)["CTC"].mean().round(2).reset_index()
    ctc_summary["Status"] = ctc_summary["AttritionFlag"].map({0: "Active", 1: "Left"})
    fig_ctc = px.bar(ctc_summary, x="Status", y="CTC", color="Status", text="CTC",
                     title="Average CTC of Active vs Ex-Employees")
    st.plotly_chart(fig_ctc, use_container_width=True)

    # ==========================
    # Step 5: Export Report
    # ==========================
    st.subheader("📄 Step 3 — Export Attrition Summary Report")
    html_summary = f"""
    <h2>Attrition Summary Report</h2>
    <p><b>Total Employees:</b> {total_employees:,}<br>
    <b>Attrition Count:</b> {total_attrition:,}<br>
    <b>Attrition Rate:</b> {attrition_rate:.2f}%<br>
    <b>Average Tenure:</b> {avg_tenure:.1f} years<br>
    <b>Average CTC:</b> ₹{avg_ctc:.2f} LPA</p>
    """
    render_pdf_download_button("Attrition Summary Report", html_summary, "Attrition_Report")

    st.markdown("""
    <hr style="border:1px solid #991B1B;margin-top:40px;"/>
    <div style="text-align:center;color:#9CA3AF;font-size:13px;">
        Prepared with ❤️ by <b>Amlan Mishra</b> | © 2025 HR Tech Portfolio
    </div>
    """, unsafe_allow_html=True)