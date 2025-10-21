# ============================================
# modules/attrition_module.py — v2.2 | Fixed Export + Stable Navigation
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_attrition_module():
    # Header
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#7F1D1D,#DC2626);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">📉 Attrition Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze turnover, tenure trends, and exit patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1: Download Template
    sample = pd.DataFrame({
        "EmployeeID": ["E1001", "E1002"],
        "Department": ["Finance", "IT"],
        "JobLevel": ["Analyst", "Manager"],
        "Gender": ["Male", "Female"],
        "TenureMonths": [24, 60],
        "AttritionFlag": ["Yes", "No"],
        "ExitReason": ["Better Pay", ""],
        "CTC": [600000, 1200000]
    })
    render_download_template("Attrition Data Template", sample, "Attrition_Template.csv")

    # Step 2: Upload
    df = upload_data("Upload Attrition Data (CSV/Excel)")
    if df is None:
        st.info("Upload dataset to continue.")
        return

    # Validation
    required = ["EmployeeID", "Department", "JobLevel", "Gender", "TenureMonths", "AttritionFlag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    df["AttritionFlag"] = df["AttritionFlag"].astype(str).str.lower().map(
        {"yes": "Yes", "y": "Yes", "1": "Yes", "no": "No", "n": "No", "0": "No"}
    )
    st.dataframe(df.head(), use_container_width=True)

    # Metrics
    total = len(df)
    left = (df["AttritionFlag"] == "Yes").sum()
    rate = (left / total * 100) if total else 0
    avg_tenure = df["TenureMonths"].mean()

    st.metric("Overall Attrition Rate", f"{rate:.1f}%")
    st.metric("Average Tenure (months)", f"{avg_tenure:.1f}")

    # Department Trends
    dept_summary = df.groupby("Department", observed=True)["AttritionFlag"].apply(lambda x: (x == "Yes").mean() * 100).reset_index(name="AttritionRate")
    fig1 = px.bar(dept_summary, x="Department", y="AttritionRate", text="AttritionRate",
                  color="Department", title="Attrition % by Department")
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    # Tenure Cohort
    df["TenureCohort"] = pd.cut(df["TenureMonths"], bins=[0, 12, 36, 60, 120],
                                labels=["<1yr", "1–3yrs", "3–5yrs", "5+yrs"])
    cohort = df.groupby("TenureCohort", observed=True)["AttritionFlag"].apply(lambda x: (x == "Yes").mean() * 100).reset_index(name="AttritionRate")
    fig2 = px.bar(cohort, x="TenureCohort", y="AttritionRate", text="AttritionRate",
                  color="TenureCohort", title="Attrition by Tenure Cohort")
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # Exit Reasons
    if "ExitReason" in df.columns:
        rc = df[df["AttritionFlag"] == "Yes"]["ExitReason"].value_counts().reset_index()
        rc.columns = ["Reason", "Count"]
        fig3 = px.pie(rc, values="Count", names="Reason", title="Top Exit Reasons")
        st.plotly_chart(fig3, use_container_width=True)

    # Export Report
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")

    data_blocks = [
        {"title": "Attrition Overview", "desc": "Turnover metrics by department and tenure.", "df": dept_summary,
         "insights": [f"Overall rate: {rate:.1f}%", f"Avg tenure: {avg_tenure:.1f} months", f"Total left: {left}/{total}"]},
        {"title": "Tenure Cohorts", "desc": "Attrition rates by tenure band.", "df": cohort,
         "insights": ["Shorter tenure = higher turnover tendency."]}
    ]

    render_pdf_download_button(
        report_title="Attrition Analytics Executive Report",
        module_name="Attrition",
        data_blocks=data_blocks,
        filename_prefix="Attrition"
    )