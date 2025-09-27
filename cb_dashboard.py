# cb_dashboard.py — Compensation & Benefits Dashboard (Hybrid Mode, with clickable PDF TOC)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from datetime import datetime
import os

# ReportLab imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm

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
# Sample CSV Templates
# -----------------------
sample_cols = ["EmployeeID","Department","JobRole","JobLevel","CTC","Bonus","Gender","PerformanceRating"]
sample_csv = pd.DataFrame(columns=sample_cols).to_csv(index=False)

benchmark_cols = ["JobRole","CTC"]
benchmark_csv = pd.DataFrame(columns=benchmark_cols).to_csv(index=False)

col1, col2 = st.columns(2)
with col1:
    st.download_button("📥 Download Sample Compensation CSV", data=sample_csv,
                       file_name="sample_compensation_template.csv", mime="text/csv")
with col2:
    st.download_button("📥 Download Sample Benchmark CSV", data=benchmark_csv,
                       file_name="sample_benchmark_template.csv", mime="text/csv")

# -----------------------
# Upload Section
# -----------------------
uploaded_file = st.file_uploader("📂 Upload Compensation Dataset (CSV/XLSX)", type=["csv","xlsx"])
benchmark_file = st.file_uploader("📂 Upload Benchmarking Dataset (CSV/XLSX)", type=["csv","xlsx"])

if not uploaded_file:
    st.info("⬆️ Please upload your Compensation dataset to begin.")
    st.stop()

# -----------------------
# Normalize Columns
# -----------------------
def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        "department": "Department",
        "jobrole": "JobRole", "job role": "JobRole",
        "joblevel": "JobLevel", "job level": "JobLevel",
        "ctc": "CTC", "salary": "CTC", "monthlyincome": "CTC",
        "bonus": "Bonus",
        "gender": "Gender",
        "performance": "PerformanceRating", "rating": "PerformanceRating"
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
# Hybrid Chart Helper
# -----------------------
def hybrid_chart(fig_plotly, fig_mpl, filename):
    st.plotly_chart(fig_plotly, use_container_width=True)
    p = os.path.join(TMP_DIR, filename)
    fig_mpl.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig_mpl)

# -----------------------
# Metrics
# -----------------------
st.subheader("📊 Key Compensation Metrics")
sections, images, tables = [], [], []

# Avg CTC by Department
if "Department" in df.columns and "CTC" in df.columns:
    avg_ctc_dept = df.groupby("Department")["CTC"].mean().reset_index()
    st.markdown("### 🏢 Average CTC by Department")
    st.dataframe(avg_ctc_dept, use_container_width=True)
    fig_p = px.bar(avg_ctc_dept, x="Department", y="CTC", color="Department", title="Avg CTC by Department")
    fig_m, ax = plt.subplots(figsize=(8,4)); sns.barplot(data=avg_ctc_dept, x="Department", y="CTC", ax=ax); plt.xticks(rotation=30)
    hybrid_chart(fig_p, fig_m, "avg_ctc_dept.png")
    sections.append(("Average CTC by Department", "Shows average compensation across departments."))
    images.append((os.path.join(TMP_DIR,"avg_ctc_dept.png"), "Avg CTC by Department"))
    tables.append(("Avg CTC by Department", avg_ctc_dept))

# Avg CTC by Job Role
if "JobRole" in df.columns and "CTC" in df.columns:
    avg_ctc_role = df.groupby("JobRole")["CTC"].mean().reset_index()
    st.markdown("### 👔 Average CTC by Job Role")
    st.dataframe(avg_ctc_role, use_container_width=True)
    fig_p = px.bar(avg_ctc_role, x="JobRole", y="CTC", color="JobRole", title="Avg CTC by Job Role")
    fig_m, ax = plt.subplots(figsize=(8,4)); sns.barplot(data=avg_ctc_role, x="JobRole", y="CTC", ax=ax); plt.xticks(rotation=45)
    hybrid_chart(fig_p, fig_m, "avg_ctc_role.png")
    sections.append(("Average CTC by Job Role", "Shows average compensation across job roles."))
    images.append((os.path.join(TMP_DIR,"avg_ctc_role.png"), "Avg CTC by Job Role"))
    tables.append(("Avg CTC by Job Role", avg_ctc_role))

# Quartile Analysis
if "JobRole" in df.columns and "CTC" in df.columns:
    q_summary = df.groupby("JobRole")["CTC"].describe(percentiles=[0.25,0.5,0.75]).reset_index()
    st.markdown("### 📐 Quartile Placement Analysis")
    st.dataframe(q_summary, use_container_width=True)
    fig_p = px.box(df, x="JobRole", y="CTC", title="Quartile Placement Analysis")
    fig_m, ax = plt.subplots(figsize=(10,6)); sns.boxplot(data=df, x="JobRole", y="CTC", ax=ax); plt.xticks(rotation=45)
    hybrid_chart(fig_p, fig_m, "quartile_analysis.png")
    sections.append(("Quartile Analysis", "Shows pay distribution within job roles and quartile placement."))
    images.append((os.path.join(TMP_DIR,"quartile_analysis.png"), "Quartile Placement Analysis"))
    tables.append(("Quartile Summary", q_summary))

# Bonus %
if "Bonus" in df.columns and "CTC" in df.columns:
    df["BonusPct"] = (df["Bonus"]/df["CTC"])*100
    bonus_summary = df.groupby("Department")["BonusPct"].mean().reset_index()
    st.markdown("### 🎁 Avg Bonus % of CTC by Department")
    st.dataframe(bonus_summary, use_container_width=True)
    fig_p = px.bar(bonus_summary, x="Department", y="BonusPct", color="Department", title="Bonus % by Department")
    fig_m, ax = plt.subplots(figsize=(8,4)); sns.barplot(data=bonus_summary, x="Department", y="BonusPct", ax=ax); plt.xticks(rotation=30)
    hybrid_chart(fig_p, fig_m, "bonus_pct.png")
    sections.append(("Bonus % of CTC", "Shows average bonus percentage across departments."))
    images.append((os.path.join(TMP_DIR,"bonus_pct.png"), "Bonus % by Department"))
    tables.append(("Bonus % by Department", bonus_summary))

# Performance vs Compensation
if "PerformanceRating" in df.columns and "CTC" in df.columns:
    perf_summary = df.groupby("PerformanceRating")["CTC"].mean().reset_index()
    st.markdown("### ⭐ Performance vs Compensation")
    st.dataframe(perf_summary, use_container_width=True)
    fig_p = px.bar(perf_summary, x="PerformanceRating", y="CTC", color="PerformanceRating", title="CTC by Rating")
    fig_m, ax = plt.subplots(figsize=(6,4)); sns.barplot(data=perf_summary, x="PerformanceRating", y="CTC", ax=ax)
    hybrid_chart(fig_p, fig_m, "perf_vs_ctc.png")
    sections.append(("Performance vs Compensation", "Shows how pay varies with performance ratings."))
    images.append((os.path.join(TMP_DIR,"perf_vs_ctc.png"), "Performance vs Compensation"))
    tables.append(("Performance vs Compensation", perf_summary))

# Benchmark
if benchmark_df is not None and "JobRole" in df.columns and "CTC" in df.columns:
    company_median = df.groupby("JobRole")["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
    market_median = benchmark_df.groupby("JobRole")["CTC"].median().reset_index().rename(columns={"CTC":"MarketMedian"})
    compare = pd.merge(company_median, market_median, on="JobRole", how="inner")
    st.markdown("### 📊 Company vs Market Benchmarking (Median CTC)")
    st.dataframe(compare, use_container_width=True)
    fig_p = px.bar(compare, x="JobRole", y=["CompanyMedian","MarketMedian"], barmode="group", title="Company vs Market Median")
    fig_m, ax = plt.subplots(figsize=(10,6)); compare.set_index("JobRole")[["CompanyMedian","MarketMedian"]].plot(kind="bar", ax=ax); plt.xticks(rotation=45)
    hybrid_chart(fig_p, fig_m, "benchmark_compare.png")
    sections.append(("Company vs Market", "Comparison of company vs market median CTC."))
    images.append((os.path.join(TMP_DIR,"benchmark_compare.png"), "Company vs Market Median"))
    tables.append(("Company vs Market", compare))

# -----------------------
# PDF Builder
# -----------------------
def draw_background(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#f5f7fa"))
    canvas.rect(0,0,A4[0],A4[1],stroke=0,fill=1)
    canvas.restoreState()

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(280, 15, f"Page {doc.page}")
    canvas.restoreState()

def build_pdf(title, sections, images, tables, master=True):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []

    # Cover
    story.append(Spacer(1, 30))
    cover_style = ParagraphStyle("Cover", parent=styles["Title"], fontSize=24, alignment=1,
                                 textColor=colors.HexColor("#0b5cff"))
    story.append(Paragraph(title, cover_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}", styles["Normal"]))
    story.append(PageBreak())

    # TOC
    if master:
        story.append(Paragraph("Table of Contents", styles["Heading2"]))
        for i, (stitle, _) in enumerate(sections, 1):
            story.append(Paragraph(f'<a href="#sec{i}">{i}. {stitle}</a>', styles["Normal"]))
        story.append(PageBreak())

    # Sections
    for i, (stitle, stext) in enumerate(sections, 1):
        story.append(Paragraph(f'<a name="sec{i}"/>{stitle}', styles["Heading2"]))
        story.append(Paragraph(stext, styles["BodyText"]))
        story.append(Spacer(1, 8))
    story.append(PageBreak())

    # Tables
    for ttitle, df in tables:
        story.append(Paragraph(ttitle, styles["Heading3"]))
        data = [list(df.columns)] + df.head(40).values.tolist()
        tstyle = TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#ddeeff")),
                             ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                             ("ALIGN",(0,0),(-1,-1),"CENTER")])
        for i in range(1,len(data)):
            if i%2==0:
                tstyle.add("BACKGROUND",(0,i),(-1,i),colors.whitesmoke)
        tbl = Table(data, repeatRows=1); tbl.setStyle(tstyle)
        story.append(tbl); story.append(PageBreak())

    # Images
    for path, caption in images:
        if os.path.exists(path):
            story.append(Paragraph(caption, styles["Heading3"]))
            story.append(RLImage(path, width=170*mm, height=90*mm))
            story.append(PageBreak())

    def on_page(canvas, doc):
        draw_background(canvas, doc)
        add_page_number(canvas, doc)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    pdf_bytes = buf.getvalue(); buf.close()
    return pdf_bytes

# -----------------------
# Exports
# -----------------------
st.subheader("📥 Download Reports & Images")

col1, col2 = st.columns(2)

with col1:
    if st.button("📑 Download Master PDF Report"):
        pdf = build_pdf("Compensation & Benefits Report", sections, images, tables, master=True)
        st.download_button("⬇️ Master PDF", data=pdf, file_name="cb_master_report.pdf", mime="application/pdf")

with col2:
    st.markdown("### 📊 Download Images (for PPT)")
    download_map = {img[1]: img[0] for img in images}
    for label, path in download_map.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(f"⬇️ {label}", f.read(), file_name=os.path.basename(path), mime="image/png")