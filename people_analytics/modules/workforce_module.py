# ============================================
# modules/workforce_module.py — v1.2 | Universal Upload + PDF Export
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report
from utils.uploader_helper import upload_data

def run_workforce_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#0B5E3D,#10B981);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">🏢 Workforce & Talent Planning</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze structure, spans, and skill inventory.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📤 Step 1 — Upload Workforce Data")
    df = upload_data("Upload Workforce Data")
    if df is None:
        return

    headcount = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount")
    fig = px.bar(headcount, x="JobLevel", y="Headcount", text="Headcount", title="Headcount by Level")
    st.plotly_chart(fig, use_container_width=True)

    data_blocks = [{
        "title": "Workforce Overview",
        "desc": "Headcount and job-level summary.",
        "df": headcount,
        "insights": ["Workforce structure balanced across job levels."]
    }]
    export_module_report("Workforce Analytics Executive Report","Workforce",data_blocks,"Workforce")