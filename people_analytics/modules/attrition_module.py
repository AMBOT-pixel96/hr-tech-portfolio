# ============================================
# modules/attrition_module.py — v2.0 | Enhanced Metrics + Executive PDF Export
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report


def run_attrition_module():
    # =========================
    # 📉 Header
    # =========================
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#7F1D1D,#DC2626);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">📉 Attrition Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze employee turnover, identify high-risk segments, and uncover tenure-based attrition trends.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📄 Step 1 — Download Template
    # =========================
    st.subheader("📄 Step 1 — Download Attrition Data Template")

    sample_data = pd.DataFrame({
        "EmployeeID": ["E1001", "E1002", "E1003"],
        "Department": ["Finance", "IT", "HR"],
        "JobLevel": ["Analyst", "Manager", "Executive"],
        "Gender": ["Male", "Female", "Female"],
        "TenureMonths": [24, 60, 12],
        "AttritionFlag": ["Yes", "No", "Yes"],
        "ExitReason": ["Better Pay", "", "Relocation"],
        "CTC": [600000, 1200000, 450000]
    })

    render_download_template("Attrition Data Template", sample_data, "Attrition_Template.csv")

    # =========================
    # 📤 Step 2 — Upload Data
    # =========================
    st.subheader("📤 Step 2 — Upload Attrition Dataset")
    uploaded = st.file_uploader(
        "Upload Attrition Data (CSV, Excel, or Text)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )

    if not uploaded:
        st.info("Please upload your attrition dataset to continue.")
        return

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, engine="openpyxl")
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # =========================
    # 🧮 Validation
    # =========================
    required = ["EmployeeID", "Department", "JobLevel", "Gender", "TenureMonths", "AttritionFlag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Normalize Attrition Flag
    df["AttritionFlag"] = df["AttritionFlag"].astype(str).str.strip().str.lower().replace({
        "yes": "Yes", "y": "Yes", "1": "Yes",
        "no": "No", "n": "No", "0": "No"
    })

    st.dataframe(df.head(), use_container_width=True)

    # =========================
    # 📊 Step 3 — Core Metrics
    # =========================
    st.subheader("📊 Attrition Insights")

    total_employees = len(df)
    total_left = (df["AttritionFlag"] == "Yes").sum()
    turnover_rate = (total_left / total_employees * 100) if total_employees > 0 else 0
    avg_tenure = df["TenureMonths"].mean()

    st.metric("Overall Attrition Rate", f"{turnover_rate:.1f}%")
    st.metric("Average Tenure (months)", f"{avg_tenure:.1f}")

    # Department-wise Attrition
    dept_summary = (
        df.groupby("Department", observed=True)["AttritionFlag"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="AttritionRate")
    )
    dept_highest = dept_summary.loc[dept_summary["AttritionRate"].idxmax(), "Department"]
    fig_dept = px.bar(dept_summary, x="Department", y="AttritionRate", text="AttritionRate",
                      title="Attrition % by Department", color="Department",
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig_dept.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_dept, use_container_width=True)

    # Job Level Analysis
    job_summary = (
        df.groupby("JobLevel", observed=True)["AttritionFlag"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="AttritionRate")
    )
    fig_job = px.bar(job_summary, x="JobLevel", y="AttritionRate", text="AttritionRate",
                     title="Attrition % by Job Level", color="JobLevel")
    fig_job.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_job, use_container_width=True)

    # Tenure Cohorts
    st.subheader("⏳ Tenure Cohort Analysis")
    df["TenureCohort"] = pd.cut(
        df["TenureMonths"], bins=[0, 12, 36, 60, 120],
        labels=["<1 year", "1–3 years", "3–5 years", "5+ years"], include_lowest=True
    )
    tenure_summary = (
        df.groupby("TenureCohort", observed=True)["AttritionFlag"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="AttritionRate")
    )
    fig_tenure = px.bar(tenure_summary, x="TenureCohort", y="AttritionRate", text="AttritionRate",
                        title="Attrition by Tenure Cohort", color="TenureCohort")
    fig_tenure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_tenure, use_container_width=True)

    # Exit Reasons (if available)
    if "ExitReason" in df.columns:
        reason_counts = df[df["AttritionFlag"] == "Yes"]["ExitReason"].value_counts().reset_index()
        reason_counts.columns = ["ExitReason", "Count"]
        fig_reason = px.pie(reason_counts, values="Count", names="ExitReason",
                            title="Top Exit Reasons")
        st.plotly_chart(fig_reason, use_container_width=True)

    # =========================
    # 💡 Insights
    # =========================
    st.markdown(f"""
    <div style="background:#0F172A;padding:10px 15px;border-radius:8px;margin-top:10px;">
    <b>💡 Key Insights:</b><br>
    • Overall attrition rate: <b>{turnover_rate:.1f}%</b><br>
    • Highest attrition department: <b>{dept_highest}</b><br>
    • Average tenure: <b>{avg_tenure:.1f} months</b><br>
    • Total employees left: <b>{total_left}</b> of <b>{total_employees}</b>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📄 Step 4 — Export Executive Report
    # =========================
    st.markdown("---")
    st.subheader("📄 Step 4 — Export Executive Report")

    data_blocks = [
        {
            "title": "Attrition Overview",
            "desc": "Turnover rate, department-level attrition, and job-level distribution.",
            "df": dept_summary,
            "insights": [
                f"Overall attrition rate: {turnover_rate:.1f}%",
                f"Highest attrition department: {dept_highest}",
                f"Average tenure: {avg_tenure:.1f} months"
            ]
        },
        {
            "title": "Tenure Cohort Insights",
            "desc": "Attrition rates across different tenure groups.",
            "df": tenure_summary,
            "insights": [
                "Shorter tenure groups (<1 year) often show higher turnover.",
                "Longer tenure tends to correlate with stability."
            ]
        }
    ]

    export_module_report(
        report_title="Attrition Analytics Executive Report",
        module_name="Attrition",
        data_blocks=data_blocks,
        filename_prefix="Attrition"
    )