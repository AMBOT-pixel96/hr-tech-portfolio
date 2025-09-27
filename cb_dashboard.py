# cb_dashboard.py -- Compensation & Benefits Dashboard (with Benchmarking)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
import io

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="💰 Compensation & Benefits Dashboard", layout="wide")
st.title("💰 Compensation & Benefits Dashboard")

# ===============================
# FILE UPLOAD
# ===============================
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("📂 Upload Compensation Dataset (CSV/XLSX)", type=["csv","xlsx"])

sample_cols = ["EmployeeID","Department","JobRole","JobLevel","Gender","CTC","Bonus","PerformanceRating","MarketMedian"]
st.sidebar.download_button(
    "📥 Download Sample Template",
    data=pd.DataFrame(columns=sample_cols).to_csv(index=False),
    file_name="compensation_template.csv",
    mime="text/csv"
)

if not uploaded_file:
    st.info("⬆️ Please upload a dataset to continue.")
    st.stop()

# Load file
if uploaded_file.name.endswith("csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.write("### 👀 Preview Uploaded Data")
st.dataframe(df.head(), use_container_width=True)

# ===============================
# METRICS
# ===============================
st.header("📊 Key Compensation Metrics")

# Avg CTC by Department
avg_ctc_dept = df.groupby("Department")["CTC"].mean().reset_index()
fig1 = px.bar(avg_ctc_dept, x="Department", y="CTC", title="Average CTC by Department", text_auto=".2s")
st.plotly_chart(fig1, use_container_width=True)

# Avg CTC by Job Role
avg_ctc_role = df.groupby("JobRole")["CTC"].mean().reset_index()
fig2 = px.bar(avg_ctc_role, x="JobRole", y="CTC", title="Average CTC by Job Role", text_auto=".2s")
st.plotly_chart(fig2, use_container_width=True)

# Quartile Placement Analysis
def quartile_flag(x, q1, q2, q3):
    if x <= q1: return "Q1 (Low)"
    elif x <= q2: return "Q2"
    elif x <= q3: return "Q3"
    else: return "Q4 (High)"

quartile_df = df.copy()
q1, q2, q3 = quartile_df["CTC"].quantile([0.25,0.5,0.75])
quartile_df["Quartile"] = quartile_df["CTC"].apply(lambda x: quartile_flag(x,q1,q2,q3))
heatmap_data = quartile_df.pivot_table(index="JobRole", columns="Quartile", values="CTC", aggfunc="count", fill_value=0)
st.write("### 📐 Quartile Distribution by Job Role")
st.dataframe(heatmap_data)

fig3 = px.imshow(heatmap_data, text_auto=True, aspect="auto", title="Quartile Heatmap by Job Role")
st.plotly_chart(fig3, use_container_width=True)

# Bonus % of CTC
df["BonusPct"] = (df["Bonus"] / df["CTC"] * 100).round(2)
bonus_dept = df.groupby("Department")["BonusPct"].mean().reset_index()
fig4 = px.bar(bonus_dept, x="Department", y="BonusPct", title="Average Bonus % of CTC by Department", text_auto=".2f")
st.plotly_chart(fig4, use_container_width=True)

bonus_gender = df.groupby("Gender")["BonusPct"].mean().reset_index()
fig5 = px.bar(bonus_gender, x="Gender", y="BonusPct", title="Average Bonus % of CTC by Gender", text_auto=".2f")
st.plotly_chart(fig5, use_container_width=True)

# Performance x Compensation
perf_ctc = df.groupby(["PerformanceRating","Department"])["CTC"].mean().reset_index()
fig6 = px.bar(perf_ctc, x="PerformanceRating", y="CTC", color="Department", barmode="group",
              title="Avg CTC by Performance Rating (Dept-wise)")
st.plotly_chart(fig6, use_container_width=True)

# ===============================
# BENCHMARKING
# ===============================
if "MarketMedian" in df.columns:
    st.header("📊 Benchmarking: Company vs Market Medians")

    bench = df.groupby("JobRole")[["CTC","MarketMedian"]].median().reset_index()
    fig7 = px.bar(
        bench, x="JobRole", y=["CTC","MarketMedian"], 
        barmode="group", title="Company Median vs Market Median (by Job Role)"
    )
    st.plotly_chart(fig7, use_container_width=True)

    bench["Delta"] = (bench["CTC"] - bench["MarketMedian"]).round(2)
    st.write("### 🔍 Benchmarking Table")
    st.dataframe(bench)

# ===============================
# PDF EXPORT
# ===============================
st.header("📑 Export Report")
if st.button("📥 Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(200,10,"Compensation & Benefits Report", ln=True, align="C")

    pdf.set_font("Arial","",12)
    pdf.multi_cell(0,10,"This report contains key metrics on employee compensation, quartiles, bonuses, performance linkage, and market benchmarking.")

    pdf.set_font("Arial","B",12)
    pdf.cell(0,10,"Company vs Market Medians (Sample)", ln=True)
    pdf.set_font("Arial","",10)
    if "MarketMedian" in df.columns:
        for _,row in bench.iterrows():
            pdf.cell(0,8,f"{row['JobRole']}: Company {row['CTC']:.2f}, Market {row['MarketMedian']:.2f}, Δ {row['Delta']:.2f}", ln=True)

    buf = io.BytesIO()
    pdf.output(buf)
    st.download_button(
        "📥 Download PDF",
        data=buf.getvalue(),
        file_name="Compensation_Report.pdf",
        mime="application/pdf"
    )

# ===============================
# CSV EXPORT
# ===============================
st.header("📦 Export Data")
st.download_button(
    "📥 Download Quartile Analysis CSV",
    data=quartile_df.to_csv(index=False),
    file_name="quartile_analysis.csv",
    mime="text/csv"
)