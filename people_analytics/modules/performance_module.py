import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import save_chart_image  # ✅ new

MODULE_COLOR = "#2563EB"

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

    dept_summary = _round_df(df.groupby("Department", observed=True)["PerformanceRating"]
                             .agg(["mean","median","count","std"]).reset_index()
                             .rename(columns={"mean":"MeanRating","median":"MedianRating","count":"Count","std":"StdDev"}))
    job_summary = _round_df(df.groupby("JobLevel", observed=True)["PerformanceRating"]
                            .agg(["mean","median","count"]).reset_index()
                            .rename(columns={"mean":"MeanRating","median":"MedianRating","count":"Count"}))
    gender_summary = _round_df(df.groupby("Gender", observed=True)["PerformanceRating"]
                               .agg(["mean","count"]).reset_index()
                               .rename(columns={"mean":"MeanRating","count":"Count"}))

    # Plots + saved images
    box_dept = px.box(df, x="Department", y="PerformanceRating", color="Department", title="Performance Ratings by Department", template="plotly_white")
    box_ctc_by_rating = px.box(df, x="PerformanceRating", y="CTC", color="PerformanceRating", title="CTC by Rating", template="plotly_white")
    kde_fig = None
    kde_path = None
    x = df["PerformanceRating"].dropna()
    if len(x) > 3:
        kde = gaussian_kde(x)
        x_range = np.linspace(max(x.min(), 0), x.max(), 200)
        kde_df = pd.DataFrame({"Rating": x_range, "Density": kde(x_range)})
        kde_fig = px.line(kde_df, x="Rating", y="Density", title="Rating Distribution (KDE)", template="plotly_white")

    dept_path = save_chart_image("Performance by Department", box_dept)
    pay_path = save_chart_image("Performance vs Pay", box_ctc_by_rating)
    if kde_fig:
        kde_path = save_chart_image("Performance Distribution", kde_fig)

    # Display
    st.subheader("Department Performance Summary")
    st.dataframe(dept_summary, use_container_width=True)
    st.plotly_chart(box_dept, use_container_width=True)
    st.subheader("Performance vs Pay")
    st.dataframe(job_summary, use_container_width=True)
    st.plotly_chart(box_ctc_by_rating, use_container_width=True)
    if kde_fig:
        st.subheader("Rating Distribution")
        st.plotly_chart(kde_fig, use_container_width=True)

    data_blocks = [
        {"title": "Performance Distribution", "desc": "Average & std-based performance spread.",
         "df": dept_summary, "fig_path": kde_path,
         "insights": [f"Avg Rating: {avg_rating:.2f}", f"StdDev: {rating_std:.2f}", f"Top performers ≥4: {top_perf_share:.1f}%"]},
        {"title": "Department Ratings", "desc": "Department-level performance analysis.",
         "df": dept_summary, "fig_path": dept_path,
         "insights": [f"Top dept: {dept_summary.sort_values('MeanRating', ascending=False).iloc[0]['Department'] if not dept_summary.empty else 'N/A'}"]},
        {"title": "Performance vs Pay", "desc": "CTC distribution across rating levels.",
         "df": job_summary, "fig_path": pay_path,
         "insights": [f"Avg CTC: ₹{avg_ctc:,.0f}"]},
        {"title": "Gender Performance", "desc": "Average performance by gender.",
         "df": gender_summary, "fig_path": None,
         "insights": [f"Top gender: {gender_summary.sort_values('MeanRating', ascending=False).iloc[0]['Gender'] if not gender_summary.empty else 'N/A'}"]}
    ]

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Performance Analytics Executive Report", "Performance", data_blocks, "Performance")