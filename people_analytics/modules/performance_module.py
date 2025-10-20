# ============================================
# modules/performance_module.py — v1.2 | PDF Export + Insight Summaries
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from utils.pdf_helper import render_pdf_download_button

def run_performance_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">🏆 Performance Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">Analyze employee performance patterns, identify high-potential clusters, and understand how ratings relate to pay and skills.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📤 Step 1 — Upload Performance Data")
perf_file = st.file_uploader(
    "Upload Performance Data (CSV, Excel, or Text)",
    type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
)
    if perf_file is None:
        st.info("Please upload a dataset to continue.")
        return

    try:
        if perf_file.name.endswith(".csv"):
            df = pd.read_csv(perf_file)
        else:
            df = pd.read_excel(perf_file, engine="openpyxl")
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender", "PerformanceRating", "CTC"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.dataframe(df.head(), use_container_width=True)

    # --- Analysis Sections ---
    with st.expander("📈 A. Performance Distribution Insights", expanded=True):
        bell_fig = px.histogram(df, x="PerformanceRating", nbins=5, color_discrete_sequence=["#60A5FA"])
        st.plotly_chart(bell_fig, use_container_width=True)

        dept_fig = px.box(df, x="Department", y="PerformanceRating", color="Department")
        st.plotly_chart(dept_fig, use_container_width=True)

        gender_avg = df.groupby("Gender", observed=True)["PerformanceRating"].mean().reset_index()
        gender_fig = px.bar(gender_avg, x="Gender", y="PerformanceRating", color="Gender", text="PerformanceRating")
        gender_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(gender_fig, use_container_width=True)

        dept_best = df.groupby("Department", observed=True)["PerformanceRating"].mean().idxmax()
        gender_diff = gender_avg["PerformanceRating"].max() - gender_avg["PerformanceRating"].min() if len(gender_avg) > 1 else 0

        st.markdown(f"""
        <div style="background:#0F172A;padding:10px 15px;border-radius:8px;margin-top:10px;">
        <b>🧠 Insights:</b><br>
        • Top-performing department: <b>{dept_best}</b><br>
        • Gender gap in ratings: <b>{gender_diff:.2f}</b> points<br>
        • Overall average rating: <b>{df['PerformanceRating'].mean():.2f}</b>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("💰 B. Performance vs Pay", expanded=True):
        pay_fig = px.box(df, x="PerformanceRating", y="CTC", color="PerformanceRating")
        st.plotly_chart(pay_fig, use_container_width=True)

        perf_pay_avg = df.groupby("PerformanceRating", observed=True)["CTC"].mean().reset_index()
        perf_pay_avg["CTC (₹ Lakhs)"] = (perf_pay_avg["CTC"] / 1e5).round(2)
        st.dataframe(perf_pay_avg[["PerformanceRating", "CTC (₹ Lakhs)"]], use_container_width=True)

        top_ctc = perf_pay_avg["CTC (₹ Lakhs)"].max()
        top_rating = perf_pay_avg.loc[perf_pay_avg["CTC (₹ Lakhs)"].idxmax(), "PerformanceRating"]

        st.markdown(f"""
        <div style="background:#0F172A;padding:10px 15px;border-radius:8px;margin-top:10px;">
        <b>💡 Insights:</b><br>
        • Highest-paying performance tier: <b>Rating {top_rating}</b><br>
        • Average CTC: <b>{top_ctc:.2f} LPA</b>
        </div>
        """, unsafe_allow_html=True)

    # --- PDF Export ---
    st.subheader("📄 Step 3 — Export Summary Report")
    html_summary = f"""
    <h2>Performance Analytics Summary</h2>
    <p>This report summarizes the performance distribution and pay correlation insights.</p>
    <div class='summary'>
    <p><b>Top Department:</b> {dept_best}<br>
    <b>Gender Gap:</b> {gender_diff:.2f} points<br>
    <b>Highest Paying Tier:</b> Rating {top_rating} ({top_ctc:.2f} LPA)</p>
    </div>
    """
    render_pdf_download_button("Performance Analytics Report", html_summary, "Performance_Report")

    st.markdown("""
    <hr style="border:1px solid #1E3A8A;margin-top:40px;"/>
    <div style="text-align:center;color:#9CA3AF;font-size:13px;">
        Prepared with ❤️ by <b>Amlan Mishra</b> | © 2025 HR Tech Portfolio
    </div>
    """, unsafe_allow_html=True)