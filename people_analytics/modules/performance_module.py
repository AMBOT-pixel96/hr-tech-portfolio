# ============================================
# modules/performance_module.py — v1.3 | KDE Curve + Universal Upload + PDF Export
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from utils.pdf_auto_exporter import export_module_report
from utils.uploader_helper import upload_data

def run_performance_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">🏆 Performance Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze employee performance patterns, identify high-potential clusters,
            and understand how ratings relate to pay and skills.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📤 Step 1 — Upload Performance Data")
    df = upload_data("Upload Performance Data (CSV or Excel)")
    if df is None:
        st.info("Please upload a dataset to continue.")
        return

    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender", "PerformanceRating", "CTC"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.dataframe(df.head(), use_container_width=True)

    # --- A. Performance Distribution Insights ---
    with st.expander("📈 A. Performance Distribution Insights", expanded=True):
        x = df["PerformanceRating"].dropna()
        kde = gaussian_kde(x)
        x_range = np.linspace(x.min(), x.max(), 200)
        y_values = kde(x_range)

        bell_fig = go.Figure()
        bell_fig.add_trace(go.Scatter(
            x=x_range, y=y_values,
            mode="lines",
            line=dict(color="#60A5FA", width=3),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.3)",
            name="Distribution"
        ))
        bell_fig.update_layout(
            title="Performance Rating Bell Curve",
            xaxis_title="Performance Rating",
            yaxis_title="Density",
            template="plotly_dark",
            showlegend=False
        )
        st.plotly_chart(bell_fig, use_container_width=True)

        dept_fig = px.box(df, x="Department", y="PerformanceRating", color="Department")
        st.plotly_chart(dept_fig, use_container_width=True)

        gender_avg = df.groupby("Gender", observed=True)["PerformanceRating"].mean().reset_index()
        gender_fig = px.bar(gender_avg, x="Gender", y="PerformanceRating", color="Gender", text="PerformanceRating")
        gender_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(gender_fig, use_container_width=True)

        dept_best = df.groupby("Department", observed=True)["PerformanceRating"].mean().idxmax()
        gender_diff = gender_avg["PerformanceRating"].max() - gender_avg["PerformanceRating"].min()

        st.markdown(f"""
        <div style="background:#0F172A;padding:10px 15px;border-radius:8px;">
        <b>🧠 Insights:</b><br>
        • Top-performing department: <b>{dept_best}</b><br>
        • Gender gap: <b>{gender_diff:.2f}</b> points<br>
        • Average rating: <b>{df['PerformanceRating'].mean():.2f}</b>
        </div>
        """, unsafe_allow_html=True)

    # --- B. Performance vs Pay ---
    with st.expander("💰 B. Performance vs Pay", expanded=True):
        pay_fig = px.box(df, x="PerformanceRating", y="CTC", color="PerformanceRating")
        st.plotly_chart(pay_fig, use_container_width=True)

        perf_pay_avg = df.groupby("PerformanceRating", observed=True)["CTC"].mean().reset_index()
        perf_pay_avg["CTC (₹ Lakhs)"] = (perf_pay_avg["CTC"] / 1e5).round(2)
        st.dataframe(perf_pay_avg, use_container_width=True)

        top_ctc = perf_pay_avg["CTC (₹ Lakhs)"].max()
        top_rating = perf_pay_avg.loc[perf_pay_avg["CTC (₹ Lakhs)"].idxmax(), "PerformanceRating"]

    # --- Export ---
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")

    data_blocks = [{
        "title": "Performance Overview",
        "desc": "Distribution, Pay correlation, and Rating Insights.",
        "df": perf_pay_avg,
        "insights": [
            f"Top-performing department: {dept_best}",
            f"Gender gap: {gender_diff:.2f}",
            f"Highest-paying rating tier: {top_rating} ({top_ctc:.2f} LPA)"
        ]
    }]

    export_module_report("Performance Analytics Executive Report", "Performance", data_blocks, "Performance")