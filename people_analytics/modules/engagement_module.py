# ============================================
# modules/engagement_module.py — v2.2
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_engagement_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💬 Engagement Analytics</h2>
        <p>Analyze employee engagement and identify hotspots.</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Engagement Survey File")
    if df is None:
        return

    question_cols = [col for col in df.columns if col.startswith("Q")]
    df["EngagementIndex"] = df[question_cols].mean(axis=1)
    dept = df.groupby("Department", observed=True)["EngagementIndex"].mean().reset_index()

    fig = px.bar(dept, x="Department", y="EngagementIndex", text="EngagementIndex",
                 color="Department", title="Average Engagement by Department")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    data_blocks = [{"title": "Engagement Summary", "desc": "Engagement score by department.",
                    "df": dept, "insights": ["High engagement in key departments."]}]

    render_pdf_download_button("Engagement Analytics Executive Report", "Engagement", data_blocks, "Engagement")