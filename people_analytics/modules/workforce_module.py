# ============================================
# modules/workforce_module.py — v2.2
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_workforce_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#0B5E3D,#10B981);
                color:white; text-align:center; margin-bottom:20px;">
        <h2>🏢 Workforce Analytics</h2>
        <p>Analyze structure, job-level headcount, and balance.</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Workforce Data")
    if df is None: return

    hc = df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount")
    fig = px.bar(hc, x="JobLevel", y="Headcount", text="Headcount", color="JobLevel", title="Headcount by Job Level")
    st.plotly_chart(fig, use_container_width=True)

    data_blocks = [{"title": "Workforce Summary", "desc": "Headcount overview by job level.",
                    "df": hc, "insights": ["Balanced structure across levels."]}]

    render_pdf_download_button("Workforce Analytics Executive Report", "Workforce", data_blocks, "Workforce")