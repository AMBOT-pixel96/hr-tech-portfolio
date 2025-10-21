# ============================================
# modules/performance_module.py — v2.2
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_performance_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2>🏆 Performance Analytics</h2>
        <p>Analyze ratings, pay linkage, and gender patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Performance Data")
    if df is None: return

    required = ["EmployeeID", "Department", "JobLevel", "Gender", "PerformanceRating", "CTC"]
    if not all(c in df.columns for c in required):
        st.error(f"Missing required columns: {', '.join(required)}")
        return

    st.dataframe(df.head(), use_container_width=True)

    # Bell Curve
    x = df["PerformanceRating"].dropna()
    kde = gaussian_kde(x)
    x_range = np.linspace(x.min(), x.max(), 200)
    st.line_chart(pd.DataFrame({"Rating": x_range, "Density": kde(x_range)}).set_index("Rating"))

    gender_avg = df.groupby("Gender", observed=True)["PerformanceRating"].mean().reset_index()
    fig = px.bar(gender_avg, x="Gender", y="PerformanceRating", color="Gender", text="PerformanceRating",
                 title="Average Rating by Gender")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    data_blocks = [{"title": "Performance Summary", "desc": "Ratings and pay correlation.",
                    "df": gender_avg, "insights": ["Consistent performance patterns observed."]}]

    render_pdf_download_button("Performance Analytics Executive Report", "Performance", data_blocks, "Performance")