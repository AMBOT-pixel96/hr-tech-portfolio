# modules/performance_module.py — v2.9 | Executive (aligned with PDF v3.1)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

MODULE_COLOR = "#2563EB"  # performance blue

def _round_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float","int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2

def run_performance_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#1E3A8A,#2563EB);color:white;">
      <h2 style="margin:0">🏆 Performance Analytics</h2>
      <p style="margin:4px 0 0 0;">Distribution, department variance & pay correlation (Executive view).</p>
    </div>
    """, unsafe_allow_html=True)

    df = upload_data("Upload Performance Data (CSV/XLSX)")
    if df is None:
        return

    required = ["EmployeeID","Department","JobLevel","Gender","PerformanceRating","CTC"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # clean types
    df["PerformanceRating"] = pd.to_numeric(df["PerformanceRating"], errors="coerce")
    df["CTC"] = pd.to_numeric(df["CTC"], errors="coerce")

    # KPIs
    avg_rating = float(df["PerformanceRating"].mean())
    rating_std = float(df["PerformanceRating"].std())
    avg_ctc = float(df["CTC"].mean())
    top_perf_share = float((df["PerformanceRating"] >= 4).mean() * 100)
    low_perf_share = float((df["PerformanceRating"] <= 2).mean() * 100)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Avg Rating", f"{avg_rating:.2f}")
    c2.metric("Rating StdDev", f"{rating_std:.2f}")
    c3.metric("Avg CTC", f"₹{avg_ctc:,.0f}")
    c4.metric("Top Performers (≥4)", f"{top_perf_share:.1f}%")
    c5.metric("Low Performers (≤2)", f"{low_perf_share:.1f}%")

    # summaries (aggregated)
    dept_summary = df.groupby("Department", observed=True)["PerformanceRating"].agg(["mean","median","count","std"]).reset_index()
    dept_summary.columns = ["Department","MeanRating","MedianRating","Count","StdDev"]
    dept_summary = _round_df(dept_summary)

    job_summary = df.groupby("JobLevel", observed=True)["PerformanceRating"].agg(["mean","median","count"]).reset_index()
    job_summary.columns = ["JobLevel","MeanRating","MedianRating","Count"]
    job_summary = _round_df(job_summary)

    gender_summary = df.groupby("Gender", observed=True)["PerformanceRating"].agg(["mean","count"]).reset_index()
    gender_summary.columns = ["Gender","MeanRating","Count"]
    gender_summary = _round_df(gender_summary)

    # Figures: ensure white template for PDF export (keeps palette colors)
    # Boxplot: rating by department
    box_dept = px.box(df, x="Department", y="PerformanceRating", title="Performance Ratings by Department", color="Department")
    box_dept.update_layout(template="plotly_white")
    box_dept.update_traces(marker_line_color='black', marker_line_width=1)

    # Performance vs Pay: CTC distribution by rating
    box_ctc_by_rating = px.box(df, x="PerformanceRating", y="CTC", title="CTC distribution by Performance Rating", color="PerformanceRating")
    box_ctc_by_rating.update_layout(template="plotly_white")
    box_ctc_by_rating.update_traces(marker_line_color='black', marker_line_width=1)

    # KDE (if enough points)
    kde_fig = None
    x = df["PerformanceRating"].dropna()
    if len(x) > 3:
        try:
            kde = gaussian_kde(x)
            x_range = np.linspace(max(x.min(), 0), x.max(), 200)
            y = kde(x_range)
            kde_df = pd.DataFrame({"Rating": x_range, "Density": y})
            kde_fig = px.line(kde_df, x="Rating", y="Density", title="Performance Rating Distribution (KDE)")
            kde_fig.update_layout(template="plotly_white")
        except Exception:
            kde_fig = None

    # App display
    st.subheader("Department Performance Summary")
    st.dataframe(dept_summary, use_container_width=True)
    st.plotly_chart(box_dept, use_container_width=True)

    st.subheader("Performance vs Pay")
    st.dataframe(job_summary, use_container_width=True)
    st.plotly_chart(box_ctc_by_rating, use_container_width=True)

    st.subheader("Rating Distribution")
    if kde_fig:
        st.plotly_chart(kde_fig, use_container_width=True)
    else:
        st.info("Not enough rating points for KDE visualization.")

    # prepare data blocks for PDF (one block per metric)
    data_blocks = [
        {
            "title": "Performance Distribution",
            "desc": "Distribution summary: average, std, top/low shares.",
            "df": dept_summary,
            "fig": kde_fig,
            "insights": [
                f"Average rating: {avg_rating:.2f}",
                f"Rating StdDev: {rating_std:.2f}",
                f"Top performers (>=4): {top_perf_share:.1f}%",
                f"Low performers (<=2): {low_perf_share:.1f}%"
            ]
        },
        {
            "title": "Department Ratings",
            "desc": "Mean and variation of ratings per department.",
            "df": dept_summary,
            "fig": box_dept,
            "insights": [
                f"Top department: {dept_summary.sort_values('MeanRating', ascending=False).iloc[0]['Department'] if not dept_summary.empty else 'N/A'}"
            ]
        },
        {
            "title": "Performance vs Pay",
            "desc": "CTC distribution across rating tiers.",
            "df": job_summary,
            "fig": box_ctc_by_rating,
            "insights": [
                f"Average CTC: ₹{avg_ctc:,.0f}"
            ]
        },
        {
            "title": "Gender Performance",
            "desc": "Mean ratings by gender.",
            "df": gender_summary,
            "fig": None,
            "insights": [
                f"Top gender by mean rating: {gender_summary.sort_values('MeanRating', ascending=False).iloc[0]['Gender'] if not gender_summary.empty else 'N/A'}"
            ]
        }
    ]

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Performance Analytics Executive Report", "Performance", data_blocks, "Performance")