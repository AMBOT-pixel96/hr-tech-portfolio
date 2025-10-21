# ============================================
# modules/engagement_module.py — v2.0 | Universal Upload + PDF Export
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report
from utils.uploader_helper import upload_data

def run_engagement_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💬 Engagement Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze employee engagement scores and identify hotspots.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📤 Step 1 — Upload Engagement Survey")
    df = upload_data("Upload Survey File")
    if df is None:
        return

    question_cols = [col for col in df.columns if col.startswith("Q")]
    df["EngagementIndex"] = df[question_cols].mean(axis=1)
    dept_summary = df.groupby("Department", observed=True)["EngagementIndex"].mean().reset_index()

    fig = px.bar(dept_summary, x="Department", y="EngagementIndex", text="EngagementIndex",
                 title="Average Engagement by Department")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    data_blocks = [{
        "title": "Engagement Summary",
        "desc": "Average engagement scores by department.",
        "df": dept_summary,
        "insights": ["High engagement seen in top-performing departments."]
    }]
    export_module_report("Engagement Analytics Executive Report","Engagement",data_blocks,"Engagement")