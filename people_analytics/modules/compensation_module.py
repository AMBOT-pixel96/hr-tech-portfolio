# ============================================
# modules/compensation_module.py — v2.2 | Fixed Export + Upload Validation
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_compensation_module():
    # Header
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#14532D,#22C55E);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💰 Compensation Analytics</h2>
        <p>Analyze pay, bonus, gender gap, and market comparison.</p>
    </div>
    """, unsafe_allow_html=True)

    # Templates
    emp_sample = pd.DataFrame([{"EmployeeID": "E1001", "Gender": "Male", "Department": "Finance",
                                "JobRole": "Analyst", "JobLevel": "Analyst", "CTC": 600000,
                                "Bonus": 50000, "PerformanceRating": 3}])
    bench_sample = pd.DataFrame([{"JobRole": "Analyst", "JobLevel": "Analyst", "MarketMedianCTC": 650000}])

    c1, c2 = st.columns(2)
    with c1: render_download_template("Internal Template", emp_sample, "Internal_Template.csv")
    with c2: render_download_template("Benchmark Template", bench_sample, "Benchmark_Template.csv")

    # Upload
    emp_df = upload_data("Upload Internal Data")
    bench_df = upload_data("Upload Benchmark Data (optional)")
    if emp_df is None:
        st.info("Upload internal file to begin.")
        return

    required = ["EmployeeID", "Gender", "Department", "JobRole", "JobLevel", "CTC", "Bonus", "PerformanceRating"]
    missing = [c for c in required if c not in emp_df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    emp_df["BonusPct"] = (emp_df["Bonus"] / emp_df["CTC"]) * 100
    st.dataframe(emp_df.head(), use_container_width=True)

    avg_ctc = emp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index()
    avg_bonus = emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index()
    gender_gap = emp_df.groupby("Gender", observed=True)["CTC"].mean().reset_index()

    # Benchmark Comparison
    if bench_df is not None and all(col in bench_df.columns for col in ["JobRole", "JobLevel", "MarketMedianCTC"]):
        merged = emp_df.merge(bench_df, on=["JobRole", "JobLevel"], how="left")
        merged["DiffPct"] = ((merged["CTC"] - merged["MarketMedianCTC"]) / merged["MarketMedianCTC"]) * 100
        comp = merged.groupby("JobLevel", observed=True)[["CTC", "MarketMedianCTC", "DiffPct"]].mean().reset_index()
        fig = px.bar(comp, x="JobLevel", y="DiffPct", text="DiffPct", color="JobLevel", title="Company vs Market Median (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")

    data_blocks = [
        {"title": "Compensation Overview", "desc": "Pay, bonus, gender gap, market benchmarks.",
         "df": avg_ctc, "insights": ["Gender gap and market differences analyzed."]}
    ]

    render_pdf_download_button("Compensation Analytics Executive Report", "Compensation", data_blocks, "Compensation")