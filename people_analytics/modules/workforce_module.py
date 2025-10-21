# ============================================
# modules/workforce_module.py — v2.0 | Structure, Spans & Skills + Executive PDF Export
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report


def run_workforce_module():
    # =========================
    # 🏢 Header
    # =========================
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#0B5E3D,#10B981);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">🏢 Workforce & Talent Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Understand your organization’s structure, manager spans, and skill distribution
            to optimize headcount and planning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📄 Step 1 — Download Template
    # =========================
    st.subheader("📄 Step 1 — Download Workforce Template")
    sample = pd.DataFrame([{
        "EmployeeID": "E1001", "ManagerID": "M001", "Department": "Finance",
        "JobLevel": "Analyst", "JobRole": "Analyst", "Gender": "Male",
        "TenureMonths": 24, "CTC": 600000, "Skills": "Excel, PowerBI"
    }])
    render_download_template("Workforce Data Template", sample, "Workforce_Template.csv")

    # =========================
    # 📤 Step 2 — Upload Data
    # =========================
    st.subheader("📤 Step 2 — Upload Workforce Dataset")
    wf_file = st.file_uploader(
        "Upload Workforce Data (CSV, Excel, or Text)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )

    if not wf_file:
        st.info("Please upload your workforce dataset to continue.")
        return

    try:
        if wf_file.name.endswith(".csv"):
            df = pd.read_csv(wf_file)
        else:
            df = pd.read_excel(wf_file, engine="openpyxl")
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    required = ["EmployeeID", "Department", "JobLevel", "JobRole"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.dataframe(df.head(), use_container_width=True)

    # =========================
    # 📊 Step 3 — Headcount & Structure
    # =========================
    st.subheader("📊 Headcount Distribution & Org Pyramid")

    headcount = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount")
    fig_pyramid = px.bar(
        headcount, x="Headcount", y="JobLevel", orientation="h",
        title="Headcount by Job Level", text="Headcount",
        color="JobLevel"
    )
    st.plotly_chart(fig_pyramid, use_container_width=True)

    # =========================
    # 🧭 Step 4 — Manager Spans
    # =========================
    st.subheader("🧭 Manager Span Analysis")
    if "ManagerID" in df.columns:
        mgr_counts = df.groupby("ManagerID", observed=True)["EmployeeID"].nunique().reset_index(name="DirectReports")
        avg_span = mgr_counts["DirectReports"].mean()
        st.metric("Average Direct Reports per Manager", f"{avg_span:.2f}")
        fig_span = px.histogram(mgr_counts, x="DirectReports", nbins=20, title="Distribution of Manager Spans")
        st.plotly_chart(fig_span, use_container_width=True)
    else:
        avg_span = 0
        st.warning("ManagerID column missing. Span analysis skipped.")

    # =========================
    # 🧠 Step 5 — Skills Inventory
    # =========================
    st.subheader("🧠 Skill Inventory Analysis")
    if "Skills" in df.columns:
        skills = (
            df["Skills"].fillna("")
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )
        skills_count = skills.value_counts().reset_index()
        skills_count.columns = ["Skill", "Count"]
        fig_skills = px.bar(skills_count.head(15), x="Skill", y="Count", text="Count",
                            title="Top 15 Skills in Workforce", color="Skill")
        st.plotly_chart(fig_skills, use_container_width=True)
        top_skill = skills_count.iloc[0]["Skill"]
    else:
        skills_count = pd.DataFrame()
        top_skill = "N/A"
        st.warning("Skills column missing. Skipping skill analysis.")

    # =========================
    # 📄 Step 6 — Export Executive Report
    # =========================
    st.markdown("---")
    st.subheader("📄 Step 6 — Export Executive Report")

    data_blocks = [
        {
            "title": "Workforce Structure Overview",
            "desc": "Headcount distribution by job level and overall span of control.",
            "df": headcount,
            "insights": [
                f"Average manager span: {avg_span:.2f}",
                "Pyramid structure visualized by job level."
            ]
        },
        {
            "title": "Skill Inventory Insights",
            "desc": "Frequency distribution of workforce skillsets.",
            "df": skills_count.head(20),
            "insights": [
                f"Top skill observed: {top_skill}",
                "Skill diversity indicates areas of development opportunity."
            ]
        }
    ]

    export_module_report(
        report_title="Workforce Analytics Executive Report",
        module_name="Workforce & Talent",
        data_blocks=data_blocks,
        filename_prefix="Workforce"
    )