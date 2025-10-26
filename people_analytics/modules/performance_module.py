# ============================================
# modules/performance_module.py — v3.0.1 | Executive Stable (Dual Theme + Sync Save)
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button
from utils.chart_saver import save_chart_image
from utils.fix_helper import ensure_chart_saved

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

    dept_summary = df.groupby("Department", observed=True)["PerformanceRating"].agg(["mean","median","count","std"]).reset_index()
    dept_summary.columns = ["Department","MeanRating","MedianRating","Count","StdDev"]
    dept_summary = _round_df(dept_summary)

    job_summary = df.groupby("JobLevel", observed=True)["PerformanceRating"].agg(["mean","median","count"]).reset_index()
    job_summary.columns = ["JobLevel","MeanRating","MedianRating","Count"]
    job_summary = _round_df(job_summary)

    gender_summary = df.groupby("Gender", observed=True)["PerformanceRating"].agg(["mean","count"]).reset_index()
    gender_summary.columns = ["Gender","MeanRating","Count"]
    gender_summary = _round_df(gender_summary)

    box_dept = px.box(df, x="Department", y="PerformanceRating", color="Department", title="Performance Ratings by Department")
    box_ctc_by_rating = px.box(df, x="PerformanceRating", y="CTC", color="PerformanceRating", title="CTC distribution by Rating")
    kde_fig = None

    x = df["PerformanceRating"].dropna()
    if len(x) > 3:
        kde = gaussian_kde(x)
        x_range = np.linspace(max(x.min(), 0), x.max(), 200)
        y = kde(x_range)
        kde_df = pd.DataFrame({"Rating": x_range, "Density": y})
        kde_fig = px.line(kde_df, x="Rating", y="Density", title="Performance Rating Distribution (KDE)")

    st.subheader("Department Performance Summary")
    st.dataframe(dept_summary, use_container_width=True)
    st.plotly_chart(box_dept, use_container_width=True)

    st.subheader("Performance vs Pay")
    st.dataframe(job_summary, use_container_width=True)
    st.plotly_chart(box_ctc_by_rating, use_container_width=True)

    st.subheader("Rating Distribution")
    if kde_fig:
        st.plotly_chart(kde_fig, use_container_width=True)

    data_blocks = [
        {"title":"Performance Distribution","desc":"KDE & distribution","df":dept_summary,"fig":kde_fig,
         "insights":[f"Avg rating {avg_rating:.2f}",f"Top ≥4: {top_perf_share:.1f}%"]},
        {"title":"Department Ratings","desc":"Boxplot by department","df":dept_summary,"fig":box_dept,
         "insights":[f"Top dept: {dept_summary.sort_values('MeanRating', ascending=False).iloc[0]['Department']}"]},
        {"title":"Performance vs Pay","desc":"CTC vs rating","df":job_summary,"fig":box_ctc_by_rating,
         "insights":[f"Avg CTC: ₹{avg_ctc:,.0f}"]},
        {"title":"Gender Performance","desc":"Mean ratings by gender","df":gender_summary,"fig":None,
         "insights":[f"Top gender: {gender_summary.sort_values('MeanRating', ascending=False).iloc[0]['Gender']}"]}
    ]

    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    render_pdf_download_button("Performance Analytics Executive Report", "Performance", data_blocks, "Performance")
# ============================================
# ➕ Add to Consolidated Leadership Deck (Performance)
# ============================================
import os, shutil, json
from utils_consolidated.pdf_merger import TMP_DIR
from utils_consolidated.deck_state_tracker import update_module_state

st.markdown("---")
st.subheader("🧩 Add to Consolidated Leadership Deck")

module_name = "Performance"
pdf_filename = f"{module_name}_Analytics_Executive_Report.pdf"

possible_paths = [
    os.path.join("/tmp", pdf_filename),
    os.path.join(os.getcwd(), pdf_filename)
]

existing_pdf = next((p for p in possible_paths if os.path.exists(p)), None)
dest_path = os.path.join(TMP_DIR, f"{module_name}.pdf")

# --- If already added ---
if os.path.exists(dest_path):
    st.success("✅ A copy of this report has been added to the consolidated deck queue.")

else:
    if existing_pdf:
        if st.button(f"➕ Add {module_name} Report to Consolidated Deck", use_container_width=True):
            try:
                shutil.copyfile(existing_pdf, dest_path)
                update_module_state(module_name)
                st.success("✅ A copy of this report has been added to the consolidated deck queue.")

                # 🔹 Auto-write metadata JSON for consolidated summary
                meta = {
                    "insights": f"Avg Rating {avg_rating:.2f} • Top Performers ≥4: {top_perf_share:.1f}% • Avg CTC ₹{avg_ctc:,.0f}",
                    "metrics_short": "Avg Rating, Top Performers %, Avg CTC"
                }
                with open(os.path.join(TMP_DIR, "Performance.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f)

            except Exception as e:
                st.error(f"⚠️ Failed to add report: {e}")
    else:
        st.info("⚙️ Generate the PDF first before adding to the consolidated deck.")