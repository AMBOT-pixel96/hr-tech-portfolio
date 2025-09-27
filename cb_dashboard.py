# app.py -- Compensation & Benefits Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io, os
from datetime import datetime

# ==============================
# Config
# ==============================
st.set_page_config(page_title="💰 C&B Dashboard", layout="wide")
st.title("💰 Compensation & Benefits Dashboard")

# ==============================
# File Uploads
# ==============================
uploaded_file = st.file_uploader("📂 Upload Employee Dataset (CSV/Excel)", type=["csv", "xlsx"])
benchmark_file = st.file_uploader("📂 Upload Market Benchmark File (optional, CSV)", type=["csv"])

sample_headers = ["EmployeeID","Department","JobRole","JobLevel","Gender","CTC","Bonus","PerfRating"]
st.download_button(
    "📥 Download Sample Employee Template",
    data=pd.DataFrame(columns=sample_headers).to_csv(index=False),
    file_name="cb_dashboard_template.csv"
)

benchmark_headers = ["JobRole","MarketMedianCTC"]
st.download_button(
    "📥 Download Sample Benchmark Template",
    data=pd.DataFrame(columns=benchmark_headers).to_csv(index=False),
    file_name="benchmark_template.csv"
)

if not uploaded_file:
    st.info("⬆️ Upload your Employee dataset to get started.")
    st.stop()

# Load employee file
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.write("### 👀 Preview Employee Data")
st.dataframe(df.head(), use_container_width=True)

# Load benchmark file if provided
benchmark_df = None
if benchmark_file:
    benchmark_df = pd.read_csv(benchmark_file)
    st.write("### 📊 Preview Benchmark Data")
    st.dataframe(benchmark_df.head(), use_container_width=True)

# ==============================
# Metrics Calculations
# ==============================
st.header("📊 Key Metrics")

# 1. Avg CTC by Department
avg_ctc_dept = df.groupby("Department")["CTC"].mean().reset_index()

# 2. Avg CTC by Job Role
avg_ctc_role = df.groupby("JobRole")["CTC"].mean().reset_index()

# 3. Gender Pay Gap (% difference)
gender_pay = df.groupby("Gender")["CTC"].mean()
if len(gender_pay) >= 2:
    gap = (gender_pay.max() - gender_pay.min()) / gender_pay.max() * 100
else:
    gap = np.nan

# 4. Bonus % of CTC by Dept
df["BonusPct"] = (df["Bonus"] / df["CTC"]) * 100
bonus_dept = df.groupby("Department")["BonusPct"].mean().reset_index()

# 5. Perf Rating x CTC
perf_ctc = df.groupby("PerfRating")["CTC"].mean().reset_index()

# 6. Benchmark comparison
benchmark_merge = None
if benchmark_df is not None:
    benchmark_merge = pd.merge(avg_ctc_role, benchmark_df, on="JobRole", how="left")
    benchmark_merge["GapVsMarket"] = benchmark_merge["CTC"] - benchmark_merge["MarketMedianCTC"]

# ==============================
# Visuals
# ==============================
st.header("📈 Visual Insights")

# Donut chart - Gender distribution
fig1, ax1 = plt.subplots()
ax1.pie(gender_pay, labels=gender_pay.index, autopct="%1.1f%%", startangle=90)
st.pyplot(fig1)

# Bar - Avg CTC by Dept
fig2, ax2 = plt.subplots()
sns.barplot(x="Department", y="CTC", data=avg_ctc_dept, ax=ax2)
plt.xticks(rotation=30)
st.pyplot(fig2)

# Heatmap - Quartile distribution by Job Level
quartiles = df.groupby("JobRole")["CTC"].apply(
    lambda x: pd.qcut(x, q=4, labels=["Q1","Q2","Q3","Q4"]).value_counts()
).unstack().fillna(0)
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(quartiles, annot=True, cmap="YlOrBr", fmt="g", ax=ax3)
st.pyplot(fig3)

# Benchmark barplot
if benchmark_merge is not None:
    fig4, ax4 = plt.subplots()
    benchmark_merge.plot(
        x="JobRole", y=["CTC","MarketMedianCTC"], kind="bar", ax=ax4
    )
    plt.title("Company vs Market Median (by JobRole)")
    plt.xticks(rotation=30)
    st.pyplot(fig4)

# ==============================
# Export PDF
# ==============================
class PDF(FPDF):
    def header(self):
        self.set_font("Arial","B",14)
        self.cell(0,10,"Compensation & Benefits Dashboard Report",ln=True,align="C")
    def footer(self):
        self.set_y(-20)
        self.set_font("Arial","I",8)
        self.cell(0,10,"Prepared with <3 by Amlan Mishra",align="C")

if st.button("📑 Generate PDF Report"):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial","",12)
    pdf.multi_cell(0,10,"Executive Summary:\nThis report provides insights on compensation & benefits including CTC, bonus, gender parity, and market benchmarking.")

    # Avg CTC by Dept
    pdf.ln(5)
    pdf.set_font("Arial","B",12)
    pdf.cell(0,10,"Avg CTC by Department",ln=True)
    pdf.set_font("Arial","",10)
    for _, row in avg_ctc_dept.iterrows():
        pdf.cell(0,10,f"{row['Department']}: {row['CTC']:.2f}",ln=True)

    # Benchmark summary
    if benchmark_merge is not None:
        pdf.ln(5)
        pdf.set_font("Arial","B",12)
        pdf.cell(0,10,"Benchmark Comparison",ln=True)
        pdf.set_font("Arial","",10)
        for _, row in benchmark_merge.iterrows():
            pdf.cell(0,10,f"{row['JobRole']}: Co Avg {row['CTC']:.2f} vs Market {row['MarketMedianCTC']:.2f} (Gap {row['GapVsMarket']:.2f})",ln=True)

    # Save PDF
    buf = io.BytesIO()
    pdf.output(buf)
    st.download_button("📥 Download PDF", data=buf.getvalue(), file_name="cb_dashboard_report.pdf", mime="application/pdf")