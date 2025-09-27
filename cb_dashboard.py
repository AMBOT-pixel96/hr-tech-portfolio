# cb_dashboard.py -- Compensation & Benefits Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Compensation & Benefits Dashboard", layout="wide")
st.title("💰 Compensation & Benefits Dashboard")

# ==============================
# Upload Section
# ==============================
uploaded_file = st.file_uploader("📂 Upload Compensation Dataset (CSV/XLSX)", type=["csv","xlsx"])
benchmark_file = st.file_uploader("📂 Upload Benchmarking Dataset (CSV/XLSX)", type=["csv","xlsx"])

if not uploaded_file:
    st.info("⬆️ Please upload your Compensation dataset to begin.")
    st.stop()

# ==============================
# Helper: Normalize Columns
# ==============================
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

# ==============================
# Load Main Data
# ==============================
if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

df = normalize_columns(df)

# Load Benchmark Data (optional)
benchmark_df = None
if benchmark_file:
    if benchmark_file.name.endswith("csv"):
        benchmark_df = pd.read_csv(benchmark_file)
    else:
        benchmark_df = pd.read_excel(benchmark_file)
    benchmark_df = normalize_columns(benchmark_df)

# ==============================
# Preview Data
# ==============================
st.subheader("👀 Preview Uploaded Data")
st.dataframe(df.head(), use_container_width=True)

# ==============================
# Key Metrics
# ==============================
st.subheader("📊 Key Compensation Metrics")

# Avg CTC by Department
if "Department" in df.columns and "CTC" in df.columns:
    avg_ctc_dept = df.groupby("Department")["CTC"].mean().reset_index()
    st.markdown("### 🏢 Average CTC by Department")
    st.dataframe(avg_ctc_dept, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=avg_ctc_dept, x="Department", y="CTC", ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

# Avg CTC by Job Role
if "JobRole" in df.columns and "CTC" in df.columns:
    avg_ctc_role = df.groupby("JobRole")["CTC"].mean().reset_index()
    st.markdown("### 👔 Average CTC by Job Role")
    st.dataframe(avg_ctc_role, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=avg_ctc_role, x="JobRole", y="CTC", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Quartile Analysis
if "JobRole" in df.columns and "CTC" in df.columns:
    st.markdown("### 📐 Quartile Placement Analysis")
    q_summary = df.groupby("JobRole")["CTC"].describe(percentiles=[0.25,0.5,0.75]).reset_index()
    st.dataframe(q_summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df, x="JobRole", y="CTC", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Bonus as % of CTC
if "Bonus" in df.columns and "CTC" in df.columns:
    df["BonusPct"] = (df["Bonus"] / df["CTC"]) * 100
    bonus_summary = df.groupby("Department")["BonusPct"].mean().reset_index()
    st.markdown("### 🎁 Avg Bonus % of CTC by Department")
    st.dataframe(bonus_summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=bonus_summary, x="Department", y="BonusPct", ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

# Performance x Compensation
if "PerformanceRating" in df.columns and "CTC" in df.columns:
    st.markdown("### ⭐ Performance vs Compensation")
    perf_summary = df.groupby("PerformanceRating")["CTC"].mean().reset_index()
    st.dataframe(perf_summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=perf_summary, x="PerformanceRating", y="CTC", ax=ax)
    st.pyplot(fig)

# Benchmark Comparison
if benchmark_df is not None and "JobRole" in df.columns and "CTC" in df.columns:
    st.markdown("### 📊 Company vs Market Benchmarking (Median CTC)")
    company_median = df.groupby("JobRole")["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
    market_median = benchmark_df.groupby("JobRole")["CTC"].median().reset_index().rename(columns={"CTC":"MarketMedian"})
    compare = pd.merge(company_median, market_median, on="JobRole", how="inner")
    st.dataframe(compare, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10,6))
    compare.set_index("JobRole")[["CompanyMedian","MarketMedian"]].plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ==============================
# Export Section
# ==============================
st.subheader("📥 Export Reports")
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button("⬇️ Download Processed Dataset (CSV)", convert_df(df), "processed_compensation.csv", "text/csv")

if benchmark_df is not None:
    st.download_button("⬇️ Download Benchmark Comparison (CSV)", convert_df(compare), "benchmark_comparison.csv", "text/csv")