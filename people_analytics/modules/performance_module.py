# modules/performance_module.py — v2.6
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_performance_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#1E3A8A,#3B82F6);color:white;">
      <h2 style="margin:0">🏆 Performance Analytics</h2>
      <p style="margin:4px 0 0 0;">Distribution, department variance & pay correlation.</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Performance Data (CSV/XLSX)")
    if df is None:
        return

    required = ["EmployeeID","Department","JobLevel","Gender","PerformanceRating","CTC"]
    if not all(c in df.columns for c in required):
        st.error("Missing required columns for Performance module.")
        return

    df["PerformanceRating"] = pd.to_numeric(df["PerformanceRating"], errors="coerce")
    df["CTC"] = pd.to_numeric(df["CTC"], errors="coerce")

    # KPIs
    avg_rating = round(df["PerformanceRating"].mean(),2)
    top_dept = df.groupby("Department")["PerformanceRating"].mean().idxmax()
    avg_ctc = round(df["CTC"].mean(),0)
    c1,c2,c3 = st.columns(3)
    c1.metric("Avg Rating", f"{avg_rating}")
    c2.metric("Top Dept", f"{top_dept}")
    c3.metric("Avg CTC", f"₹{avg_ctc:,.0f}")

    # KDE / Bell curve
    x = df["PerformanceRating"].dropna()
    if len(x) > 1:
        kde = gaussian_kde(x)
        x_range = np.linspace(x.min(), x.max(), 200)
        density = kde(x_range)
        bell_df = pd.DataFrame({"Rating": x_range, "Density": density})
        fig_kde = px.line(bell_df, x="Rating", y="Density", title="Performance Rating Distribution (KDE)")
        st.plotly_chart(fig_kde, use_container_width=True)
    else:
        st.info("Not enough rating data for KDE chart.")
        fig_kde = None

    # Dept boxplot and CTC vs Rating scatter
    fig_box = px.box(df, x="Department", y="PerformanceRating", color="Department", title="Ratings by Department")
    fig_scatter = px.scatter(df, x="PerformanceRating", y="CTC", color="Department", hover_data=["EmployeeID"], title="CTC vs Rating")
    st.plotly_chart(fig_box, use_container_width=True)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Summary table (avg CTC by rating)
    summary = df.groupby("PerformanceRating", observed=True)["CTC"].mean().reset_index()
    summary["CTC_₹L"] = (summary["CTC"]/1e5).round(2)

    # PDF export blocks
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    data_blocks = [
        {"title":"Performance Distribution","desc":"KDE & distribution","df":df[["EmployeeID","PerformanceRating"]].head(50),"fig":fig_kde,"insights":[f"Avg Rating: {avg_rating}"]},
        {"title":"Department Ratings","desc":"Boxplot per department","df":df[["Department","PerformanceRating"]].groupby("Department").mean().reset_index(),"fig":fig_box,"insights":[]},
        {"title":"Performance vs Pay","desc":"CTC vs Rating scatter","df":summary,"fig":fig_scatter,"insights":[f"Avg CTC: ₹{avg_ctc:,.0f}"]}
    ]
    render_pdf_download_button("Performance Analytics Executive Report","Performance",data_blocks,"Performance")