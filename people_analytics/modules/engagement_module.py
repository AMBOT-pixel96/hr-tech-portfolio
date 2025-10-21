# ============================================
# modules/engagement_module.py — v2.1 | Smart Insights + PDF Export
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report

def run_engagement_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💬 Engagement Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze employee engagement survey results across departments, job levels, and demographics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Step 1: Template ---
    st.subheader("📄 Step 1 — Download Engagement Survey Template")
    sample_data = pd.DataFrame({
        "EmployeeID": ["E1001", "E1002"],
        "Department": ["Finance", "HR"],
        "JobLevel": ["Analyst", "Manager"],
        "Gender": ["Male", "Female"],
        "Q1": [5, 4],
        "Q2": [4, 3],
        "Q3": [3, 4],
        "Q4": [5, 5],
        "Q5": [4, 4],
    })
    render_download_template("Engagement Survey Template", sample_data, "Engagement_Survey_Template.csv")

    # --- Step 2: Upload ---
    st.subheader("📤 Step 2 — Upload Completed Survey File")
    uploaded = st.file_uploader("Upload filled survey (CSV, Excel, or Text)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"])
    if not uploaded:
        st.info("Please upload the completed survey file to proceed.")
        return

    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded, engine="openpyxl")
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return
    st.dataframe(df.head(), use_container_width=True)

    # --- Step 3: Validation ---
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender"]
    question_cols = [c for c in df.columns if c.startswith("Q")]
    if any(c not in df.columns for c in required_cols) or not question_cols:
        st.error("Missing required survey columns.")
        return

    for q in question_cols:
        df[q] = pd.to_numeric(df[q], errors="coerce").clip(1, 5)
    df.dropna(subset=question_cols, how="all", inplace=True)
    if df.empty:
        st.error("No valid responses found.")
        return

    # --- Step 4: Analysis ---
    df["EngagementIndex"] = df[question_cols].mean(axis=1)
    df["EngagementCategory"] = pd.cut(
        df["EngagementIndex"], bins=[0, 2.5, 3.5, 5],
        labels=["Low", "Moderate", "High"], include_lowest=True)
    overall_index = df["EngagementIndex"].mean()

    dept_summary = df.groupby("Department", observed=True)["EngagementIndex"].mean().round(2).reset_index()
    dept_best = dept_summary.loc[dept_summary["EngagementIndex"].idxmax(), "Department"]

    st.metric("Overall Engagement Index (1–5)", f"{overall_index:.2f}")
    bar = px.bar(dept_summary, x="Department", y="EngagementIndex",
                 text="EngagementIndex", color="Department",
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(bar, use_container_width=True)

    cat_counts = df["EngagementCategory"].value_counts().reindex(["High","Moderate","Low"]).fillna(0)
    pie = px.pie(values=cat_counts.values, names=cat_counts.index, title="Engagement Category Breakdown")
    st.plotly_chart(pie, use_container_width=True)

    # --- Step 5: Export ---
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    data_blocks = [{
        "title": "Engagement Insights",
        "desc": "Summarizes engagement index distribution and categorical breakdown.",
        "df": dept_summary,
        "insights": [
            f"Highest-engaged department: {dept_best}",
            f"Overall engagement index: {overall_index:.2f}",
            f"High engagement share: {(cat_counts['High']/cat_counts.sum())*100:.1f}%"
        ],
    }]
    export_module_report("Engagement Analytics Executive Report", "Engagement", data_blocks, "Engagement")