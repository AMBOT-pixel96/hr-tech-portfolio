# ============================================
# modules/attrition_module.py — v1.2 | Universal Upload + Insights + PDF Export
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from utils.pdf_helper import render_pdf_download_button
from utils.template_helper import render_download_template

def run_attrition_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#7F1D1D,#DC2626);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">📉 Attrition Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">Monitor employee turnover patterns, identify high-risk departments, and track key retention trends.</p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📄 Step 1 — Download Template
    # =========================
    st.subheader("📄 Step 1 — Download Attrition Data Template")

    sample_data = pd.DataFrame({
        "EmployeeID": ["E1001", "E1002", "E1003"],
        "Department": ["Finance", "HR", "Operations"],
        "JobLevel": ["Analyst", "Manager", "Executive"],
        "Gender": ["Male", "Female", "Male"],
        "Age": [28, 35, 40],
        "TenureYears": [2, 5, 10],
        "AttritionFlag": ["Yes", "No", "No"]
    })
    render_download_template("Attrition Data Template", sample_data, "Attrition_Template.csv")

    # =========================
    # 📤 Step 2 — Upload Data
    # =========================
    st.subheader("📤 Step 2 — Upload Attrition Data")
    uploaded = st.file_uploader(
        "Upload Attrition Data (CSV, Excel, or Text)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )

    if not uploaded:
        st.info("Please upload your dataset to begin analysis.")
        return

    # --- Read uploaded data ---
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, engine="openpyxl")
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # =========================
    # 🧮 Step 3 — Validation
    # =========================
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender", "Age", "TenureYears", "AttritionFlag"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    df["AttritionFlag"] = df["AttritionFlag"].str.strip().str.title()
    df["AttritionFlag"] = df["AttritionFlag"].replace({"Y": "Yes", "N": "No"})

    # =========================
    # 📊 Step 4 — Overall Attrition Insights
    # =========================
    st.subheader("📈 Attrition Overview")

    total_emp = len(df)
    total_attr = len(df[df["AttritionFlag"] == "Yes"])
    attr_rate = (total_attr / total_emp) * 100 if total_emp > 0 else 0

    st.metric("Overall Attrition Rate", f"{attr_rate:.1f}%", delta=None)

    # 📊 Distribution
    pie = px.pie(df, names="AttritionFlag", title="Attrition Distribution", color_discrete_sequence=["#16A34A", "#DC2626"])
    st.plotly_chart(pie, use_container_width=True)

    # =========================
    # 🏢 Step 5 — Attrition by Department
    # =========================
    st.subheader("🏢 Attrition by Department")
    dept_summary = (
        df.groupby("Department", observed=True)["AttritionFlag"]
        .value_counts(normalize=True)
        .rename("AttritionRate")
        .mul(100)
        .reset_index()
    )
    dept_summary = dept_summary[dept_summary["AttritionFlag"] == "Yes"]
    st.dataframe(dept_summary[["Department", "AttritionRate"]].round(1), use_container_width=True)

    dept_fig = px.bar(
        dept_summary,
        x="Department",
        y="AttritionRate",
        text="AttritionRate",
        title="Department-wise Attrition Rate (%)",
        color="Department",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    dept_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(dept_fig, use_container_width=True)

    # =========================
    # 👥 Step 6 — Attrition by Job Level
    # =========================
    st.subheader("👥 Attrition by Job Level")
    level_summary = (
        df.groupby("JobLevel", observed=True)["AttritionFlag"]
        .value_counts(normalize=True)
        .rename("AttritionRate")
        .mul(100)
        .reset_index()
    )
    level_summary = level_summary[level_summary["AttritionFlag"] == "Yes"]

    fig_level = px.bar(
        level_summary,
        x="JobLevel",
        y="AttritionRate",
        text="AttritionRate",
        title="Attrition by Job Level (%)",
        color="JobLevel",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_level.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_level, use_container_width=True)

    # =========================
    # 💡 Step 7 — Insight Summary
    # =========================
    top_dept = dept_summary.loc[dept_summary["AttritionRate"].idxmax(), "Department"]
    top_rate = dept_summary["AttritionRate"].max()
    top_level = level_summary.loc[level_summary["AttritionRate"].idxmax(), "JobLevel"]

    st.markdown(f"""
    <div style="background:#0F172A;padding:10px 15px;border-radius:8px;margin-top:10px;">
    <b>💡 Insights:</b><br>
    • Highest attrition department: <b>{top_dept}</b> ({top_rate:.1f}%)<br>
    • Most impacted job level: <b>{top_level}</b><br>
    • Total attrition rate: <b>{attr_rate:.1f}%</b><br>
    • Total employees analyzed: <b>{total_emp}</b>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📄 Step 8 — Export Summary Report (PDF)
    # =========================
    st.subheader("📄 Step 8 — Export Summary Report")
    html_summary = f"""
    <h2>Attrition Analytics Summary</h2>
    <p>This report summarizes key retention insights across departments and job levels.</p>
    <div class='summary'>
    <p><b>Overall Attrition Rate:</b> {attr_rate:.1f}%</p>
    <p><b>Top Department:</b> {top_dept} ({top_rate:.1f}%)</p>
    <p><b>Most Impacted Job Level:</b> {top_level}</p>
    </div>
    """
    render_pdf_download_button("Attrition Analytics Report", html_summary, "Attrition_Report")

    st.markdown("""
    <hr style="border:1px solid #7F1D1D;margin-top:40px;"/>
    <div style="text-align:center;color:#9CA3AF;font-size:13px;">
        Prepared with ❤️ by <b>Amlan Mishra</b> | © 2025 HR Tech Portfolio
    </div>
    """, unsafe_allow_html=True)