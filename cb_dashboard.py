# cb_dashboard.py — Compensation & Benefits Dashboard (Hybrid Mode)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
import os

st.set_page_config(page_title="Compensation & Benefits Dashboard", layout="wide")

# -----------------------
# TITLE / BADGES / LINKS
# -----------------------
st.title("💰 Compensation & Benefits Dashboard")

st.markdown(
    """
    ![Python](https://img.shields.io/badge/Python-3.10-blue)
    ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
    ![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Comp%20Analytics-yellow)
    """
)

st.markdown(
    """
    **By Amlan Mishra**  
    [![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github&style=for-the-badge)](https://github.com/AMBOT-pixel96/hr-tech-portfolio)  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin&style=for-the-badge)](https://www.linkedin.com/in/amlan-mishra-7aa70894)
    """
)

TMP_DIR = "temp_charts_cb"
os.makedirs(TMP_DIR, exist_ok=True)

# -----------------------
# Sample CSV Template
# -----------------------
sample_cols = ["EmployeeID","Department","JobRole","JobLevel","CTC","Bonus","Gender","PerformanceRating"]
sample_csv = pd.DataFrame(columns=sample_cols).to_csv(index=False)

st.download_button(
    "📥 Download Sample Compensation CSV",
    data=sample_csv,
    file_name="sample_compensation_template.csv",
    mime="text/csv"
)

# -----------------------
# Sample Benchmark CSV Template
# -----------------------
benchmark_cols = ["JobRole","CTC"]
benchmark_csv = pd.DataFrame(columns=benchmark_cols).to_csv(index=False)

st.download_button(
    "📥 Download Sample Benchmark CSV",
    data=benchmark_csv,
    file_name="sample_benchmark_template.csv",
    mime="text/csv"
)

# -----------------------
# Upload Section
# -----------------------
uploaded_file = st.file_uploader("📂 Upload Compensation Dataset (CSV/XLSX)", type=["csv","xlsx"])
benchmark_file = st.file_uploader("📂 Upload Benchmarking Dataset (CSV/XLSX)", type=["csv","xlsx"])

if not uploaded_file:
    st.info("⬆️ Please upload your Compensation dataset to begin.")
    st.stop()

# -----------------------
# Helper: Normalize Columns
# -----------------------
def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        "department": "Department",
        "jobrole": "JobRole",
        "job role": "JobRole",
        "joblevel": "JobLevel",
        "job level": "JobLevel",
        "ctc": "CTC",
        "salary": "CTC",
        "monthlyincome": "CTC",
        "bonus": "Bonus",
        "gender": "Gender",
        "performance": "PerformanceRating",
        "rating": "PerformanceRating"
    }
    df.rename(columns={k.lower(): v for k, v in rename_map.items()}, inplace=True)
    return df

# Load Data
if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)
df = normalize_columns(df)

benchmark_df = None
if benchmark_file:
    if benchmark_file.name.endswith("csv"):
        benchmark_df = pd.read_csv(benchmark_file)
    else:
        benchmark_df = pd.read_excel(benchmark_file)
    benchmark_df = normalize_columns(benchmark_df)

# -----------------------
# Preview Data
# -----------------------
st.subheader("👀 Preview Uploaded Data")
st.dataframe(df.head(), use_container_width=True)

# -----------------------
# Plotly + Matplotlib Hybrid Helper
# -----------------------
def hybrid_chart(fig_plotly, fig_mpl, filename):
    """Show Plotly chart, save Matplotlib clone to temp for export"""
    st.plotly_chart(fig_plotly, use_container_width=True)
    p = os.path.join(TMP_DIR, filename)
    fig_mpl.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig_mpl)

# -----------------------
# Metrics
# -----------------------
st.subheader("📊 Key Compensation Metrics")

# Avg CTC by Department
if "Department" in df.columns and "CTC" in df.columns:
    avg_ctc_dept = df.groupby("Department")["CTC"].mean().reset_index()
    st.markdown("### 🏢 Average CTC by Department")
    st.dataframe(avg_ctc_dept, use_container_width=True)

    # Plotly
    fig_p = px.bar(avg_ctc_dept, x="Department", y="CTC", color="Department", title="Avg CTC by Department")
    # Matplotlib
    fig_m, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=avg_ctc_dept, x="Department", y="CTC", ax=ax)
    plt.xticks(rotation=30)
    hybrid_chart(fig_p, fig_m, "avg_ctc_dept.png")

# Avg CTC by Job Role
if "JobRole" in df.columns and "CTC" in df.columns:
    avg_ctc_role = df.groupby("JobRole")["CTC"].mean().reset_index()
    st.markdown("### 👔 Average CTC by Job Role")
    st.dataframe(avg_ctc_role, use_container_width=True)

    fig_p = px.bar(avg_ctc_role, x="JobRole", y="CTC", color="JobRole", title="Avg CTC by Job Role")
    fig_m, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=avg_ctc_role, x="JobRole", y="CTC", ax=ax)
    plt.xticks(rotation=45)
    hybrid_chart(fig_p, fig_m, "avg_ctc_role.png")

# Quartile Analysis
if "JobRole" in df.columns and "CTC" in df.columns:
    st.markdown("### 📐 Quartile Placement Analysis")
    q_summary = df.groupby("JobRole")["CTC"].describe(percentiles=[0.25,0.5,0.75]).reset_index()
    st.dataframe(q_summary, use_container_width=True)

    fig_p = px.box(df, x="JobRole", y="CTC", title="Quartile Placement Analysis")
    fig_m, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df, x="JobRole", y="CTC", ax=ax)
    plt.xticks(rotation=45)
    hybrid_chart(fig_p, fig_m, "quartile_analysis.png")

# Bonus as % of CTC
if "Bonus" in df.columns and "CTC" in df.columns:
    df["BonusPct"] = (df["Bonus"] / df["CTC"]) * 100
    bonus_summary = df.groupby("Department")["BonusPct"].mean().reset_index()
    st.markdown("### 🎁 Avg Bonus % of CTC by Department")
    st.dataframe(bonus_summary, use_container_width=True)

    fig_p = px.bar(bonus_summary, x="Department", y="BonusPct", color="Department", title="Bonus % by Department")
    fig_m, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=bonus_summary, x="Department", y="BonusPct", ax=ax)
    plt.xticks(rotation=30)
    hybrid_chart(fig_p, fig_m, "bonus_pct.png")

# Performance x Compensation
if "PerformanceRating" in df.columns and "CTC" in df.columns:
    st.markdown("### ⭐ Performance vs Compensation")
    perf_summary = df.groupby("PerformanceRating")["CTC"].mean().reset_index()
    st.dataframe(perf_summary, use_container_width=True)

    fig_p = px.bar(perf_summary, x="PerformanceRating", y="CTC", color="PerformanceRating",
                   title="Avg CTC by Performance Rating")
    fig_m, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=perf_summary, x="PerformanceRating", y="CTC", ax=ax)
    hybrid_chart(fig_p, fig_m, "perf_vs_ctc.png")

# Benchmark Comparison
if benchmark_df is not None and "JobRole" in df.columns and "CTC" in df.columns:
    st.markdown("### 📊 Company vs Market Benchmarking (Median CTC)")
    company_median = df.groupby("JobRole")["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
    market_median = benchmark_df.groupby("JobRole")["CTC"].median().reset_index().rename(columns={"CTC":"MarketMedian"})
    compare = pd.merge(company_median, market_median, on="JobRole", how="inner")
    st.dataframe(compare, use_container_width=True)

    fig_p = px.bar(compare, x="JobRole", y=["CompanyMedian","MarketMedian"], barmode="group",
                   title="Company vs Market Median CTC")
    fig_m, ax = plt.subplots(figsize=(10,6))
    compare.set_index("JobRole")[["CompanyMedian","MarketMedian"]].plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    hybrid_chart(fig_p, fig_m, "benchmark_compare.png")

# -----------------------
# Download images for PPT decks
# -----------------------
st.subheader("📥 Download images to plug into PPT decks")
download_map = {
    "Avg CTC by Department": "avg_ctc_dept.png",
    "Avg CTC by Job Role": "avg_ctc_role.png",
    "Quartile Analysis": "quartile_analysis.png",
    "Bonus % by Department": "bonus_pct.png",
    "Performance vs CTC": "perf_vs_ctc.png",
    "Benchmark Compare": "benchmark_compare.png"
}
for label, fn in download_map.items():
    p = os.path.join(TMP_DIR, fn)
    if os.path.exists(p):
        with open(p, "rb") as f:
            st.download_button(f"⬇️ {label}", f.read(), file_name=fn, mime="image/png")
    else:
        st.write(f"- {label}: (not available)")

# -----------------------
# Polished PDF Export
# -----------------------
class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "Compensation & Benefits Report", ln=True, align="C")

    def footer(self):
        self.set_y(-20)
        self.set_font("Arial", "I", 8)
        self.set_text_color(100, 100, 100)
        txt = "Prepared with <3 by Amlan Mishra - HR Tech, People Analytics & C&B Specialist at KPMG India"
        safe = txt.encode("latin-1", "replace").decode("latin-1")
        self.multi_cell(0, 10, safe, align="C")
        self.set_text_color(30, 100, 200)
        self.set_font("Arial", "U", 8)
        self.cell(0, 10, "Connect on LinkedIn", ln=True, align="C",
                  link="https://www.linkedin.com/in/amlan-mishra-7aa70894")

if st.button("📑 Generate Polished PDF"):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "Compensation & Benefits Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated {datetime.now().strftime('%d-%b-%Y %H:%M')}", ln=True, align="C")
    pdf.ln(12)

    exec_text = (
        "Executive Summary:\n\n"
        "This report summarizes key Compensation & Benefits metrics across your workforce. "
        "It compares company medians to market medians (if provided) and breaks down pay distribution "
        "by department, job role, performance and more.\n\n"
        "Use this report to identify pay gaps, outliers and to design equitable C&B policies.\n"
    )
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, exec_text, align="L")

    chart_list = list(download_map.values())
    for fn in chart_list:
        p = os.path.join(TMP_DIR, fn)
        if os.path.exists(p):
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, os.path.splitext(fn)[0], ln=True, align="C")
            pdf.image(p, x=20, y=40, w=170)

    buf = BytesIO()
    pdf.output(buf, "S")
    st.download_button(
        "📥 Download Polished PDF",
        data=buf.getvalue(),
        file_name="compensation_benefits_report.pdf",
        mime="application/pdf",
    )