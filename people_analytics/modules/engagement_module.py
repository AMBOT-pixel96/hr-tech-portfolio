# ============================================
# modules/engagement_module.py — v2.0 | Smart Insights + PDF Export
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report

def run_engagement_module():
    # =========================
    # 💬 Header
    # =========================
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💬 Engagement Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze employee engagement survey results across departments, job levels, and demographics. 
            Identify hot zones and improvement areas instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📄 Step 1 — Download Template
    # =========================
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

    # =========================
    # 📤 Step 2 — Upload Completed Survey
    # =========================
    st.subheader("📤 Step 2 — Upload Completed Survey File")
    uploaded = st.file_uploader(
        "Upload filled survey (CSV, Excel, or Text)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )

    if not uploaded:
        st.info("Please upload the completed survey file to proceed.")
        return

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
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender"]
    question_cols = [col for col in df.columns if col.startswith("Q")]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return
    if not question_cols:
        st.error("No survey question columns (Q1, Q2, etc.) found.")
        return

    # Clean and normalize
    for q in question_cols:
        df[q] = pd.to_numeric(df[q], errors="coerce").clip(1, 5)
    df = df.dropna(subset=question_cols, how="all")
    if df.empty:
        st.error("No valid responses found after cleaning.")
        return

    # =========================
    # 📊 Step 4 — Engagement Index
    # =========================
    st.subheader("📊 Step 4 — Engagement Index Analysis")

    df["EngagementIndex"] = df[question_cols].mean(axis=1)
    df["EngagementCategory"] = pd.cut(
        df["EngagementIndex"],
        bins=[0, 2.5, 3.5, 5],
        labels=["Low", "Moderate", "High"],
        include_lowest=True,
    )

    overall_index = df["EngagementIndex"].mean()
    st.metric("Overall Engagement Index (1–5)", f"{overall_index:.2f}")

    # --- Department Summary ---
    dept_summary = df.groupby("Department", observed=True)["EngagementIndex"].mean().round(2).reset_index()
    dept_best = dept_summary.loc[dept_summary["EngagementIndex"].idxmax(), "Department"]
    st.subheader("🏢 Engagement by Department")
    st.dataframe(dept_summary, use_container_width=True)

    bar = px.bar(
        dept_summary,
        x="Department",
        y="EngagementIndex",
        text="EngagementIndex",
        title="Average Engagement by Department",
        color="Department",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(bar, use_container_width=True)

    # --- Category Distribution ---
    st.subheader("🎯 Engagement Level Distribution")
    cat_counts = df["EngagementCategory"].value_counts().reindex(["High", "Moderate", "Low"]).fillna(0).astype(int)
    pie = px.pie(values=cat_counts.values, names=cat_counts.index, title="Engagement Category Breakdown")
    st.plotly_chart(pie, use_container_width=True)

    st.markdown(f"""
    <div style="background:#0F172A;padding:10px 15px;border-radius:8px;margin-top:10px;">
    <b>💡 Insights:</b><br>
    • Highest-engaged department: <b>{dept_best}</b><br>
    • Share of highly engaged employees: <b>{(cat_counts['High']/cat_counts.sum())*100:.1f}%</b><br>
    • Overall engagement index: <b>{overall_index:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

# ==================================
# 📄 Export Executive Report
# ==================================
st.markdown("---")
st.subheader("📄 Step 5 — Export Executive Report")

data_blocks = [
    {
        "title": "Engagement Insights",
        "desc": "Summarizes engagement index distribution and categorical breakdown.",
        "df": df.head(10) if "df" in locals() else None,
        "insights": [
            "Engagement levels across departments analyzed.",
            "Distribution of High, Moderate, and Low engagement visualized."
        ],
    }
]

from utils.pdf_auto_exporter import export_module_report
export_module_report(
    report_title="Engagement Analytics Executive Report",
    module_name="Engagement",
    data_blocks=data_blocks,
    filename_prefix="Engagement"
)