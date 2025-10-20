# ============================================
# modules/performance_module.py — v1.1 Polished + Insight Summaries
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ==============================
# Performance Analytics Module
# ==============================
def run_performance_module():
    # --- Header Banner ---
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">🏆 Performance Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">Analyze employee performance patterns, 
        identify high-potential clusters, and understand how ratings relate to pay and skills.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Upload Section ---
    st.subheader("📤 Step 1 — Upload Performance Data")
    perf_file = st.file_uploader("Upload Performance Data (CSV or Excel)", type=["csv", "xlsx"])

    if perf_file is None:
        st.info("Please upload a dataset to continue.")
        return

    # --- Read File ---
    try:
        if perf_file.name.endswith(".csv"):
            df = pd.read_csv(perf_file)
        else:
            df = pd.read_excel(perf_file, engine="openpyxl")
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # --- Validation ---
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender", "PerformanceRating", "CTC"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.dataframe(df.head(), use_container_width=True)

    # --- Metrics A: Performance Distribution Insights ---
    with st.expander("📈 A. Performance Distribution Insights", expanded=True):
        # Bell Curve
        bell_fig = px.histogram(
            df,
            x="PerformanceRating",
            nbins=5,
            color_discrete_sequence=["#60A5FA"],
            title="Performance Rating Bell Curve"
        )
        st.plotly_chart(bell_fig, use_container_width=True)

        # Rating by Department
        dept_fig = px.box(
            df,
            x="Department",
            y="PerformanceRating",
            color="Department",
            title="Performance Ratings by Department"
        )
        st.plotly_chart(dept_fig, use_container_width=True)

        # Rating by Job Level
        level_fig = px.box(
            df,
            x="JobLevel",
            y="PerformanceRating",
            color="JobLevel",
            title="Performance Ratings by Job Level"
        )
        st.plotly_chart(level_fig, use_container_width=True)

        # Rating by Gender
        gender_avg = df.groupby("Gender", observed=True)["PerformanceRating"].mean().reset_index()
        gender_fig = px.bar(
            gender_avg,
            x="Gender",
            y="PerformanceRating",
            color="Gender",
            text="PerformanceRating",
            title="Average Performance Rating by Gender",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        gender_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(gender_fig, use_container_width=True)

        # Insight Summary
        dept_best = df.groupby("Department", observed=True)["PerformanceRating"].mean().idxmax()
        gender_diff = (
            gender_avg["PerformanceRating"].max() - gender_avg["PerformanceRating"].min()
            if len(gender_avg) > 1 else 0
        )
        st.markdown(f"""
        <div style="background:#0F172A; padding:10px 15px; border-radius:8px; margin-top:10px;">
        <b>🧠 Insight Summary:</b><br>
        • Department with highest average rating: <b>{dept_best}</b><br>
        • Gender difference in ratings: <b>{gender_diff:.2f}</b> points<br>
        • Overall average rating: <b>{df['PerformanceRating'].mean():.2f}</b>
        </div>
        """, unsafe_allow_html=True)

    # --- Metrics B: Performance vs Skills (Future) ---
    with st.expander("🧠 B. Performance vs Skills (Future Module)", expanded=False):
        st.info("This section will integrate skill matrix data in v2 (Talent Insights module).")

    # --- Metrics C: Performance vs Pay ---
    with st.expander("💰 C. Performance vs Pay", expanded=True):
        pay_fig = px.box(
            df,
            x="PerformanceRating",
            y="CTC",
            color="PerformanceRating",
            title="CTC by Performance Rating"
        )
        st.plotly_chart(pay_fig, use_container_width=True)

        perf_pay_avg = df.groupby("PerformanceRating", observed=True)["CTC"].mean().reset_index()
        perf_pay_avg["CTC (₹ Lakhs)"] = (perf_pay_avg["CTC"] / 1e5).round(2)
        st.dataframe(perf_pay_avg[["PerformanceRating", "CTC (₹ Lakhs)"]], use_container_width=True)

        top_ctc = perf_pay_avg["CTC (₹ Lakhs)"].max()
        top_rating = perf_pay_avg.loc[perf_pay_avg["CTC (₹ Lakhs)"].idxmax(), "PerformanceRating"]
        st.markdown(f"""
        <div style="background:#0F172A; padding:10px 15px; border-radius:8px; margin-top:10px;">
        <b>💡 Insight Summary:</b><br>
        • Highest-paying performance tier: <b>Rating {top_rating}</b><br>
        • Average CTC at this level: <b>{top_ctc:.2f} LPA</b><br>
        • Pay increases roughly <b>{df.groupby('PerformanceRating')['CTC'].mean().pct_change().mean() * 100:.1f}%</b> between tiers.
        </div>
        """, unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("""
    <hr style="border:1px solid #1E3A8A; margin-top:40px;"/>
    <div style="text-align:center; color:#9CA3AF; font-size:13px;">
        Prepared with ❤️ by <b>Amlan Mishra</b> | © 2025 HR Tech Portfolio
    </div>
    """, unsafe_allow_html=True)