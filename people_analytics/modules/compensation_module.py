# ============================================
# modules/compensation_module.py — v1.0
# Compensation & Benefits Analytics Module
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_helper import render_pdf_download_button

def run_compensation_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💰 Compensation & Benefits Analytics</h2>
        <p style="font-size:14px;margin-top:6px;">
        Analyze employee pay distribution, gender parity, and market competitiveness to ensure fair and efficient compensation strategy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================
    # Step 1: Download Templates
    # ==========================
    st.subheader("📄 Step 1 — Download Data Templates")
    emp_sample = pd.DataFrame([{
        "EmployeeID": "E1001",
        "Gender": "Male",
        "Department": "Finance",
        "JobRole": "Analyst",
        "JobLevel": "Analyst",
        "CTC": 600000,
        "Bonus": 50000,
        "PerformanceRating": 3
    }])
    bench_sample = pd.DataFrame([{
        "JobRole": "Analyst",
        "JobLevel": "Analyst",
        "MarketMedianCTC": 650000
    }])
    render_download_template("Internal Compensation Template", emp_sample, "Internal_Template.csv")
    render_download_template("Benchmark Template", bench_sample, "Benchmark_Template.csv")

    # ==========================
    # Step 2: Upload Data
    # ==========================
    st.subheader("📤 Step 2 — Upload Data Files")
    col1, col2 = st.columns(2)
    emp_file = col1.file_uploader("Upload Internal Compensation Data", type=["csv", "xlsx"])
    bench_file = col2.file_uploader("Upload Benchmark Data (optional)", type=["csv", "xlsx"])

    if emp_file is None:
        st.info("Please upload your internal compensation data to continue.")
        return

    try:
        emp_df = pd.read_csv(emp_file) if emp_file.name.endswith(".csv") else pd.read_excel(emp_file, engine="openpyxl")
        st.success("✅ Internal data uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading internal file: {e}")
        return

    bench_df = None
    if bench_file:
        try:
            bench_df = pd.read_csv(bench_file) if bench_file.name.endswith(".csv") else pd.read_excel(bench_file, engine="openpyxl")
            st.success("✅ Benchmark data uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading benchmark file: {e}")
            return

    st.dataframe(emp_df.head(), use_container_width=True)

    # ==========================
    # Step 3: Validation
    # ==========================
    required_cols = ["EmployeeID", "Gender", "Department", "JobRole", "JobLevel", "CTC", "Bonus", "PerformanceRating"]
    missing = [c for c in required_cols if c not in emp_df.columns]
    if missing:
        st.error(f"Missing columns in internal data: {', '.join(missing)}")
        return

    # ==========================
    # Step 4: Metrics
    # ==========================
    st.subheader("📊 Compensation Insights")

    emp_df["CTC"] = pd.to_numeric(emp_df["CTC"], errors="coerce")
    emp_df["Bonus"] = pd.to_numeric(emp_df["Bonus"], errors="coerce")
    emp_df = emp_df.dropna(subset=["CTC"])

    avg_ctc = emp_df["CTC"].mean() / 1e5
    median_ctc = emp_df["CTC"].median() / 1e5

    st.metric("Average CTC (₹ Lakhs)", f"{avg_ctc:.2f}")
    st.metric("Median CTC (₹ Lakhs)", f"{median_ctc:.2f}")

    # --- CTC by Job Level
    st.markdown("#### 💼 CTC by Job Level")
    job_ctc = emp_df.groupby("JobLevel", observed=True)["CTC"].median().reset_index()
    fig_job = px.bar(job_ctc, x="JobLevel", y="CTC", text="CTC", color="JobLevel",
                     title="Median CTC by Job Level", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_job.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    st.plotly_chart(fig_job, use_container_width=True)

    # --- Bonus as % of CTC
    st.markdown("#### 💸 Bonus % of CTC")
    emp_df["BonusPct"] = (emp_df["Bonus"] / emp_df["CTC"]) * 100
    bonus_summary = emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().round(2).reset_index()
    fig_bonus = px.bar(bonus_summary, x="JobLevel", y="BonusPct", color="JobLevel", text="BonusPct",
                       title="Average Bonus % by Job Level")
    st.plotly_chart(fig_bonus, use_container_width=True)

    # --- Gender Pay Gap
    st.markdown("#### ⚖️ Gender Pay Equity")
    gender_summary = emp_df.groupby("Gender", observed=True)["CTC"].mean().round(2).reset_index()
    fig_gender = px.bar(gender_summary, x="Gender", y="CTC", color="Gender", text="CTC",
                        title="Average CTC by Gender")
    fig_gender.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    st.plotly_chart(fig_gender, use_container_width=True)

    if len(gender_summary) == 2:
        male_ctc = gender_summary.loc[gender_summary["Gender"].str.lower() == "male", "CTC"].values[0]
        female_ctc = gender_summary.loc[gender_summary["Gender"].str.lower() == "female", "CTC"].values[0]
        gap_pct = ((male_ctc - female_ctc) / male_ctc) * 100 if male_ctc else 0
    else:
        gap_pct = 0

    st.markdown(f"**Gender Pay Gap:** {gap_pct:.2f}%")

    # --- Market Comparison
    if bench_df is not None:
        st.markdown("#### 🌍 Market Comparison")
        merged = emp_df.merge(bench_df, on=["JobRole", "JobLevel"], how="left")
        merged["MarketGap"] = ((merged["CTC"] - merged["MarketMedianCTC"]) / merged["MarketMedianCTC"]) * 100
        market_summary = merged.groupby("JobLevel", observed=True)["MarketGap"].mean().round(2).reset_index()
        fig_market = px.bar(market_summary, x="JobLevel", y="MarketGap", color="JobLevel", text="MarketGap",
                            title="Internal vs Market Pay (Average % Difference)")
        fig_market.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig_market, use_container_width=True)

    # ==========================
    # Step 5: Export Report
    # ==========================
    st.subheader("📄 Step 3 — Export Compensation Report")
    html_summary = f"""
    <h2>Compensation Analytics Summary</h2>
    <div class='summary'>
    <p><b>Average CTC:</b> ₹{avg_ctc:.2f} LPA<br>
    <b>Median CTC:</b> ₹{median_ctc:.2f} LPA<br>
    <b>Gender Pay Gap:</b> {gap_pct:.2f}%</p>
    </div>
    """
    render_pdf_download_button("Compensation Analytics Report", html_summary, "Compensation_Report")

    st.markdown("""
    <hr style="border:1px solid #1E3A8A;margin-top:40px;"/>
    <div style="text-align:center;color:#9CA3AF;font-size:13px;">
        Prepared with ❤️ by <b>Amlan Mishra</b> | © 2025 HR Tech Portfolio
    </div>
    """, unsafe_allow_html=True)