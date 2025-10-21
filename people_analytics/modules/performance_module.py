# ============================================
# modules/performance_module.py — v2.0 | Bell Curve + Executive PDF Export
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from utils.pdf_auto_exporter import export_module_report


def run_performance_module():
    # =========================
    # 🎯 Header
    # =========================
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

    # =========================
    # 📤 Step 1 — Upload File
    # =========================
    st.subheader("📤 Step 1 — Upload Performance Data")

    perf_file = st.file_uploader(
        "Upload Performance Data (CSV, Excel, or Text)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )

    if perf_file is None:
        st.info("Please upload a dataset to continue.")
        return

    # =========================
    # 🧮 Load Data
    # =========================
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

    # =========================
    # 📊 A. Distribution Insights
    # =========================
    with st.expander("📈 A. Performance Distribution Insights", expanded=True):
        try:
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
                name="Performance Distribution"
            ))
            bell_fig.update_layout(
                title="Performance Rating Bell Curve",
                xaxis_title="Performance Rating",
                yaxis_title="Density",
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(bell_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render bell curve: {e}")

        dept_fig = px.box(df, x="Department", y="PerformanceRating", color="Department",
                          title="Performance Ratings by Department")
        st.plotly_chart(dept_fig, use_container_width=True)

        gender_avg = df.groupby("Gender", observed=True)["PerformanceRating"].mean().reset_index()
        gender_fig = px.bar(
            gender_avg, x="Gender", y="PerformanceRating", color="Gender",
            text="PerformanceRating", title="Average Performance Rating by Gender"
        )
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

    # =========================
    # 💰 B. Performance vs Pay
    # =========================
    with st.expander("💰 B. Performance vs Pay", expanded=True):
        pay_fig = px.box(df, x="PerformanceRating", y="CTC", color="PerformanceRating",
                         title="CTC Distribution by Performance Rating")
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

    # =========================
    # 📄 Export Executive Report
    # =========================
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")

    data_blocks = [
        {
            "title": "Performance Distribution Insights",
            "desc": "Smooth bell curve distribution and departmental spread of ratings.",
            "df": gender_avg,
            "insights": [
                f"Top-performing department: {dept_best}",
                f"Gender gap in ratings: {gender_diff:.2f} points",
                f"Average performance rating: {df['PerformanceRating'].mean():.2f}"
            ]
        },
        {
            "title": "Performance vs Pay Analysis",
            "desc": "Relationship between performance ratings and compensation levels.",
            "df": perf_pay_avg,
            "insights": [
                f"Highest-paying rating tier: {top_rating} ({top_ctc:.2f} LPA)",
                "Higher ratings generally correlate with increased pay distribution."
            ]
        }
    ]

    export_module_report(
        report_title="Performance Analytics Executive Report",
        module_name="Performance",
        data_blocks=data_blocks,
        filename_prefix="Performance"
    )