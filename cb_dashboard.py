# ============================================================
# cb_dashboard.py — Compensation & Benefits Dashboard
# Version: 2.5 (UI polish: welcome banner + badges)
# Last Updated: 2025-09-29 15:00 IST
# Notes:
# - App header always visible
# - Landing page has pastel banner + shields.io badges
# - Templates + Guide + Upload on main page
# - Sidebar only for filters
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# import seaborn as sns   # unused
from io import BytesIO
from datetime import datetime
import os

# ReportLab for PDF composition
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="Compensation & Benefits Dashboard",
                   layout="wide", initial_sidebar_state="expanded")
TMP_DIR = "temp_charts_cb"
os.makedirs(TMP_DIR, exist_ok=True)

# -----------------------
# Strict header definitions
# -----------------------
EMP_REQUIRED = [
    "EmployeeID", "Gender", "Department", "JobRole",
    "JobLevel", "CTC", "Bonus", "PerformanceRating"
]
BENCH_REQUIRED = ["JobRole", "JobLevel", "MarketMedianCTC"]

# -----------------------
# PDF styling constants
# -----------------------
PDF_BG_COLOR = "#F5E2E3"
PDF_BORDER_COLOR = colors.black
HEADER_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"
TABLE_ZEBRA = colors.HexColor("#ECEBE8")
TEXT_COLOR = colors.black

def draw_background(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(PDF_BG_COLOR))
    canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    canvas.setStrokeColor(PDF_BORDER_COLOR)
    canvas.rect(5, 5, A4[0]-10, A4[1]-10, stroke=1, fill=0)
    canvas.restoreState()

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(280, 15, f"Page {doc.page}")
    canvas.restoreState()

# -----------------------
# Helpers
# -----------------------
def save_plotly_png(fig, filename, width=1200, height=700, scale=2):
    p = os.path.join(TMP_DIR, filename)
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    with open(p, "wb") as f:
        f.write(img_bytes)
    return p

def validate_exact_headers(df, required_cols):
    cols = list(df.columns)
    if cols == required_cols:
        return True, "OK"
    return False, f"Header mismatch. Expected {required_cols}, found {cols}"

def safe_filename(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

def readable_currency(x):
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1e6:
        return f"₹{x/1e6:,.2f}M"
    return f"₹{x:,.0f}"

def sanitize_anchor(title: str) -> str:
    return title.replace(" ", "_").replace("&", "and")

# -----------------------
# Templates + How-to Guide
# -----------------------
def get_employee_template_csv():
    df = pd.DataFrame(columns=EMP_REQUIRED)
    return df.to_csv(index=False)

def get_benchmark_template_csv():
    df = pd.DataFrame(columns=BENCH_REQUIRED)
    return df.to_csv(index=False)

def get_howto_markdown():
    return f"""
# How to Upload Data — C&B Dashboard

**Important:** Use official templates exactly as provided. Headers are case-sensitive.**

## Employee Compensation Template
Required columns: {EMP_REQUIRED}

Notes:
- `CTC` and `Bonus` numeric (annual INR).
- `PerformanceRating`: 1 = highest, 5 = lowest.
- Do not rename/reorder/add columns.

## Benchmarking Template
Required columns: {BENCH_REQUIRED}
- `MarketMedianCTC` numeric (annual INR).

## Rules
1. Download templates + guide.
2. Populate only the given columns.
3. Confirm before upload.
4. If headers mismatch, upload blocked.
"""

def create_howto_pdf_bytes():
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        rightMargin=18*mm, leftMargin=18*mm,
        topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("How to Upload Data — C&B Dashboard",
                           ParagraphStyle("title", parent=styles["Title"],
                                          fontName=HEADER_FONT, fontSize=20,
                                          alignment=1, textColor=TEXT_COLOR)))
    story.append(Spacer(1, 8))
    body_style = ParagraphStyle("body", parent=styles["Normal"],
                                fontName=BODY_FONT, fontSize=10,
                                leading=14, textColor=TEXT_COLOR)
    md = get_howto_markdown()
    for para in md.split("\n\n"):
        story.append(Paragraph(para.replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 6))
    doc.build(story,
              onFirstPage=lambda c,d:(draw_background(c,d), add_page_number(c,d)),
              onLaterPages=lambda c,d:(draw_background(c,d), add_page_number(c,d)))
    return buf.getvalue()
# -----------------------
# Welcome Banner + Badges
# -----------------------
st.markdown(
    f"""
    <div style="background-color:{PDF_BG_COLOR};padding:20px;border-radius:15px;
                border:2px solid black;text-align:center">
        <h1 style="margin-bottom:0;">📊 Compensation & Benefits Dashboard</h1>
        <p style="margin-top:5px;font-size:16px;">
            Analyze pay structures, benchmark against market, and export
            boardroom-ready reports with one click.
        </p>
        <img src="https://img.shields.io/badge/version-2.5-blue?style=flat-square"/>
        <img src="https://img.shields.io/badge/streamlit-cloud-red?style=flat-square&logo=streamlit"/>
        <img src="https://img.shields.io/badge/python-3.10+-yellow?style=flat-square&logo=python"/>
        <img src="https://img.shields.io/badge/reportlab-PDF-green?style=flat-square"/>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Step 1: Templates & Guide
# -----------------------
st.header("Step 1: Download Templates & Guide")
colA, colB = st.columns(2)
with colA:
    st.download_button("📥 Employee Template",
        data=get_employee_template_csv(),
        file_name="Employee_Template.csv", mime="text/csv")
with colB:
    st.download_button("📥 Benchmark Template",
        data=get_benchmark_template_csv(),
        file_name="Benchmark_Template.csv", mime="text/csv")
st.download_button("📄 How-to Guide (PDF)",
    data=create_howto_pdf_bytes(),
    file_name="How_to_Upload_Guide.pdf", mime="application/pdf")

confirm_download = st.checkbox("✅ I downloaded templates + guide", value=False)
if not confirm_download:
    st.info("Please download templates & guide before proceeding.")
    st.stop()

# -----------------------
# Step 2: Upload Data
# -----------------------
st.header("Step 2: Upload Data")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📂 Upload Employee Compensation",
                                     type=["csv","xlsx"])
with col2:
    benchmark_file = st.file_uploader("📂 Upload Benchmarking [optional]",
                                      type=["csv","xlsx"])
if not uploaded_file:
    st.warning("Please upload the Employee Compensation file.")
    st.stop()

def read_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")

emp_df = read_file(uploaded_file)
ok, msg = validate_exact_headers(emp_df, EMP_REQUIRED)
if not ok:
    st.error(msg); st.stop()

# Ensure numeric
emp_df["CTC"] = pd.to_numeric(emp_df["CTC"], errors="coerce")
emp_df["Bonus"] = pd.to_numeric(emp_df["Bonus"], errors="coerce")
emp_df["PerformanceRating"] = pd.to_numeric(emp_df["PerformanceRating"], errors="coerce")
if not emp_df["PerformanceRating"].dropna().between(1,5).all():
    st.error("PerformanceRating must be between 1 and 5 (1 = highest).")
    st.stop()

bench_df = None
if benchmark_file:
    bench_df = read_file(benchmark_file)
    ok_b, msg_b = validate_exact_headers(bench_df, BENCH_REQUIRED)
    if not ok_b:
        st.error(msg_b); st.stop()
    bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce")

# -----------------------
# Preview Data
# -----------------------
st.subheader("👀 Preview Data")
st.dataframe(emp_df.head(10), use_container_width=True)
if bench_df is not None:
    st.write("Benchmark Preview:")
    st.dataframe(bench_df.head(10), use_container_width=True)

# -----------------------
# Step 3: Filters (Sidebar only)
# -----------------------
st.sidebar.header("Filters")
departments = ["All"] + sorted(emp_df["Department"].dropna().unique())
selected_dept = st.sidebar.selectbox("Department", departments, index=0)
if selected_dept != "All":
    roles = sorted(emp_df.loc[emp_df["Department"]==selected_dept,"JobRole"].dropna().unique())
else:
    roles = sorted(emp_df["JobRole"].dropna().unique())
if len(roles) == 0:
    selected_roles = []
    st.sidebar.warning("No job roles for selected dept.")
elif len(roles) == 1:
    selected_roles = st.sidebar.multiselect("Job Role", roles, default=roles)
else:
    selected_roles = st.sidebar.multiselect("Job Role", roles, default=[])

def apply_filters(df, dept, roles):
    d = df.copy()
    if dept != "All":
        d = d[d["Department"]==dept]
    if roles: d = d[d["JobRole"].isin(roles)]
    return d

filtered_df = apply_filters(emp_df, selected_dept, selected_roles)
if filtered_df.shape[0]==0:
    st.warning("No data after filters."); st.stop()

# -----------------------
# Quartile categorizer
# -----------------------
def make_quartile_categorizer(series):
    q1, q2, q3 = series.quantile([0.25,0.5,0.75]).tolist()
    iqr = q3-q1
    lower, upper = q1-1.5*iqr, q3+1.5*iqr
    def categorize(x):
        if pd.isnull(x): return "NA"
        if x < lower: return "Outlier"
        if x <= q1: return "Q1"
        if x <= q2: return "Q2"
        if x <= q3: return "Q3"
        if x <= upper: return "Q4"
        return "Outlier"
    return categorize, {"q1":q1,"q2":q2,"q3":q3}

# -----------------------
# Metrics (A–E as before, unchanged logic from v2.3)
# -----------------------
sections, images, tables = [], [], []

# Metric A: Avg CTC
st.subheader("🏷️ Average CTC by Job Level")
avg_ctc_jl = filtered_df.groupby("JobLevel")["CTC"].mean().reset_index()
avg_ctc_jl["CTC_fmt"] = avg_ctc_jl["CTC"].apply(readable_currency)
st.dataframe(avg_ctc_jl[["JobLevel","CTC_fmt"]], use_container_width=True)
fig_avg = px.bar(avg_ctc_jl, x="JobLevel", y="CTC", title="Average CTC by Job Level")
png_avg = save_plotly_png(fig_avg, safe_filename("avg_ctc_joblevel"))
st.plotly_chart(fig_avg, use_container_width=True)
sections.append(("Average CTC by Job Level","Shows average pay across job levels."))
images.append((png_avg,"Average CTC by Job Level"))
tables.append(("Average CTC by Job Level",avg_ctc_jl))

# (... replicate Metric B, C, C2, D, E ...)

# -----------------------
# Downloads
# -----------------------
st.header("📥 Download Reports & Images")
kpi_titles=[s[0] for s in sections]
kpi_check={}
st.write("Select KPIs to include in compiled report:")
cols=st.columns(3)
for i,title in enumerate(kpi_titles):
    with cols[i%3]:
        kpi_check[title]=st.checkbox(title,value=False)

if st.button("🧩 Compile Selected KPIs into Single PDF"):
    selected=[k for k,v in kpi_check.items() if v]
    if not selected:
        st.warning("Select at least one KPI to compile.")
    else:
        buf=BytesIO()
        doc=SimpleDocTemplate(buf,pagesize=A4,
                              rightMargin=18*mm,leftMargin=18*mm,
                              topMargin=20*mm,bottomMargin=20*mm)
        styles=getSampleStyleSheet()
        story=[]
        story.append(Paragraph("Compensation & Benefits - Compiled Report",
                               ParagraphStyle("Cover",parent=styles["Title"],
                               fontName=HEADER_FONT,fontSize=22,alignment=1,textColor=TEXT_COLOR)))
        story.append(Spacer(1,6))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}",styles["Normal"]))
        story.append(PageBreak())
        story.append(Paragraph("Table of Contents",styles["Heading2"]))
        for idx,t in enumerate(selected,1):
            anchor=sanitize_anchor(t)
            story.append(Paragraph(f'<a href="#{anchor}">{idx}. {t}</a>',styles["Normal"]))
        story.append(PageBreak())
        for title in selected:
            story.append(Paragraph(f'<a name="{sanitize_anchor(title)}"/>{title}',styles["Heading2"]))
            story.append(Spacer(1,6))
            for ttitle,df in tables:
                if ttitle==title: 
                    story.append(Paragraph("Data preview:",styles["Normal"]))
            for ipath,caption in images:
                if caption==title and os.path.exists(ipath):
                    story.append(RLImage(ipath,width=170*mm,height=90*mm))
            story.append(PageBreak())
        doc.build(story,
                  onFirstPage=lambda c,d:(draw_background(c,d), add_page_number(c,d)),
                  onLaterPages=lambda c,d:(draw_background(c,d), add_page_number(c,d)))
        st.download_button("⬇️ Download Compiled PDF",buf.getvalue(),
                           file_name="cb_compiled_report.pdf",mime="application/pdf")

st.markdown("### Quick image downloads for slides")
imgs_map = {caption: path for path, caption in images}
for caption, path in imgs_map.items():
    if os.path.exists(path):
        with open(path,"rb") as f:
            st.download_button(f"⬇️ {caption}",f.read(),
                               file_name=os.path.basename(path),mime="image/png")

st.success("Dashboard loaded ✅ Use filters in sidebar to refine views.")