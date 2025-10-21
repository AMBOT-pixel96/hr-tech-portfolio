# ============================================
# modules/workforce_module.py — v1.1 | Workforce & Talent Planning + PDF Export
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report

def run_workforce_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#0B5E3D,#10B981);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">🏢 Workforce & Talent Planning</h2>
        <p style="font-size:14px; margin-top:6px;">
            Headcount, spans, pyramid, and skill inventory analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📄 Step 1 — Download Workforce Template")
    sample = pd.DataFrame([{"EmployeeID":"E1001","ManagerID":"M001","Department":"Finance","JobLevel":"Analyst","JobRole":"Analyst","Gender":"Male","TenureMonths":24,"CTC":600000,"Skills":"Excel,PowerBI"}])
    render_download_template("Workforce Data Template", sample, "Workforce_Template.csv")

    st.subheader("📤 Step 2 — Upload Workforce Data")
    wf_file = st.file_uploader("Upload Workforce Data (CSV or Excel)", type=["csv","xlsx"])
    if wf_file is None:
        st.info("Upload workforce data to continue.")
        return
    df = pd.read_csv(wf_file) if wf_file.name.endswith(".csv") else pd.read_excel(wf_file, engine="openpyxl")
    st.success("✅ File uploaded successfully!")

    headcount = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount")
    st.plotly_chart(px.bar(headcount, x="JobLevel", y="Headcount", title="Headcount by Level", text="Headcount"), use_container_width=True)

    if "ManagerID" in df.columns:
        mgr_counts = df.groupby("ManagerID", observed=True)["EmployeeID"].nunique().reset_index(name="DirectReports")
        avg_span = mgr_counts["DirectReports"].mean()
        st.metric("Average Span", f"{avg_span:.2f}")

    # --- Step 5: Export ---
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    data_blocks = [{
        "title":"Workforce Overview",
        "desc":"Headcount, spans, and skills summary.",
        "df":headcount,
        "insights":["Headcount distribution visualized.","Average managerial span calculated."]
    }]
    export_module_report("Workforce Analytics Executive Report","Workforce",data_blocks,"Workforce")