# ============================================
# modules/attrition_module.py — v2.0 | Attrition Trends + PDF Export
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
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#7F1D1D,#DC2626);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">📉 Attrition Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze employee turnover patterns, identify high-risk segments, and uncover attrition trends by department, tenure, and job level.
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
    attr_file = st.file_uploader(
        "Upload Attrition Data (CSV, Excel, or Text)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )

    if attr_file is None:
        st.info("Please upload your attrition dataset to continue.")
        return

    try:
        if attr_file.name.endswith(".csv"):
            df = pd.read_csv(attr_file)
        else:
            df = pd.read_excel(attr_file, engine="openpyxl")
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # =========================
    # 🧮 Step 3 — Validation
    # =========================
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender", "TenureMonths", "AttritionFlag"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Normalize attrition flag
    df["AttritionFlag"] = df["AttritionFlag"].astype(str).str.strip().str.lower().replace({
        "yes": "Yes", "y": "Yes", "1": "Yes",
        "no": "No", "n": "No", "0": "No"
    })

    # =========================
    # 📊 Step 4 — Core Metrics
    # =========================
    st.subheader("📊 Step 4 — Attrition Insights")

    total_employees = len(df)
    total_left = (df["AttritionFlag"] == "Yes").sum()
    turnover_rate = (total_left / total_employees * 100) if total_employees > 0 else 0

    avg_tenure = df["TenureMonths"].mean()
    st.metric("Overall Attrition Rate", f"{turnover_rate:.1f}%")
    st.metric("Average Tenure (months)", f"{avg_tenure:.1f}")

    # --- Department-wise Attrition ---
    dept_summary = (
        df.groupby("Department", observed=True)["AttritionFlag"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .round(2)
        .reset_index(name="AttritionRate")
    )
    dept_highest = dept_summary.loc[dept_summary["AttritionRate"].idxmax(), "Department"]

    st.subheader("🏢 Attrition by Department")
    st.dataframe(dept_summary, use_container_width=True)

    fig_dept = px.bar(
        dept_summary, x="Department", y="AttritionRate", text="AttritionRate",
        title="Attrition % by Department", color="Department", color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_dept.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_dept, use_container_width=True)

    # --- Job Level Analysis ---
    job_summary = (
        df.groupby("JobLevel", observed=True)["AttritionFlag"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .round(2)
        .reset_index(name="AttritionRate")
    )
    fig_job = px.bar(
        job_summary, x="JobLevel", y="AttritionRate", text="AttritionRate",
        title="Attrition % by Job Level", color="JobLevel", color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_job.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_job, use_container_width=True)

    # --- Exit Reasons ---
    if "ExitReason" in df.columns:
        exit_reason = df[df["AttritionFlag"] == "Yes"]["ExitReason"].value_counts().reset_index()
        exit_reason.columns = ["ExitReason", "Count"]
        if not exit_reason.empty:
            fig_exit = px.pie(exit_reason, values="Count", names="ExitReason", title="Top Exit Reasons")
            st.plotly_chart(fig_exit, use_container_width=True)
        else:
            st.info("No exit reasons available for visualization.")
    else:
        st.info("Exit reason column not found — skipping.")

    # =========================
    # 💡 Insights Summary
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

    
# ==================================
# 📄 Export Executive Report
# ==================================
st.markdown("---")
st.subheader("📄 Step 5 — Export Executive Report")

data_blocks = [
    {
        "title": "Attrition Summary",
        "desc": "Turnover patterns, department-level trends, and tenure-based attrition insights.",
        "df": df.head(10) if "df" in locals() else None,
        "insights": [
            "Overall attrition rate and tenure distribution summarized.",
            "Department and job-level trends highlighted for workforce planning."
        ],
    }
]

from utils.pdf_auto_exporter import export_module_report
export_module_report(
    report_title="Attrition Analytics Executive Report",
    module_name="Attrition",
    data_blocks=data_blocks,
    filename_prefix="Attrition"
)