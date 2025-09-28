# cb_dashboard.py — Compensation & Benefits Dashboard (Hybrid Mode, with clickable PDF TOC)
# Updated: Implements Step A (PDF styling) and Step B (11+ items), strict templates, dependent filters,
# per-metric image + PDF export, compile-multi-metric PDF, quartile placement, company vs market, etc.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime
import os
import base64
import tempfile
import json
import math

# ReportLab for PDF composition (used for single and combined PDFs)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="Compensation & Benefits Dashboard", layout="wide", initial_sidebar_state="collapsed")
TMP_DIR = "temp_charts_cb"
os.makedirs(TMP_DIR, exist_ok=True)

# -----------------------
# Strict header definitions (exact, case-sensitive)
# -----------------------
EMP_REQUIRED = ["EmployeeID", "Gender", "Department", "JobRole", "JobLevel", "CTC", "Bonus", "PerformanceRating"]
BENCH_REQUIRED = ["JobRole", "JobLevel", "MarketMedianCTC"]

# -----------------------
# STEP A: PDF styling constants (saved as Step A)
# -----------------------
PDF_BG_COLOR = "#F5E2E3"  # soft pastel rose
PDF_BORDER_COLOR = colors.black
HEADER_FONT = "Helvetica-Bold"
SUBHEADER_FONT = "Helvetica"
BODY_FONT = "Helvetica"  # ReportLab standard; using Helvetica variant
HIGHLIGHT_BG = colors.yellow  # for warnings
TABLE_ZEBRA = colors.HexColor("#ECEBE8")
TABLE_HEADER_BG = colors.HexColor("#FFFFFF")
TEXT_COLOR = colors.black

# -----------------------
# Utility helpers
# -----------------------
def save_plotly_png(fig, filename, width=1200, height=700, scale=2):
    """Save plotly figure to file (PNG) and return path."""
    p = os.path.join(TMP_DIR, filename)
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    with open(p, "wb") as f:
        f.write(img_bytes)
    return p

def make_download_button_bytes(data_bytes, file_name, mime, label):
    st.download_button(label, data=data_bytes, file_name=file_name, mime=mime)

def df_preview_table(df):
    st.dataframe(df, use_container_width=True)

def validate_exact_headers(df, required_cols):
    """Return (bool, message) whether df has exactly required_cols in same order."""
    cols = list(df.columns)
    if cols == required_cols:
        return True, "OK"
    else:
        return False, f"Header mismatch. Expected exact headers: {required_cols}. Found: {cols}"

def enforce_numeric_columns(df, cols):
    """Try convert to numeric and return invalids list"""
    invalid = {}
    for c in cols:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass
            nnull = df[c].isna().sum()
            invalid[c] = int(nnull)
    return invalid

def safe_filename(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

def readable_currency(x):
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1e6:
        return f"₹{x/1e6:,.2f}M"
    else:
        return f"₹{x:,.0f}"

# -----------------------
# Templates + How-to PDF (content as text markdown -> rendered in app)
# -----------------------
def get_employee_template_csv():
    df = pd.DataFrame(columns=EMP_REQUIRED)
    return df.to_csv(index=False)

def get_benchmark_template_csv():
    df = pd.DataFrame(columns=BENCH_REQUIRED)
    return df.to_csv(index=False)

def get_howto_markdown():
    md = f"""
# How to Upload Data — C&B Dashboard

**Important:** Use the official templates exactly as provided. Column headers must match exactly (case-sensitive).

## Employee Compensation Template
Required columns (must match exactly):
{EMP_REQUIRED}

**Notes:**
- `CTC` and `Bonus` should be numeric (annual INR).
- `PerformanceRating`: **1 is the highest, 5 is the lowest**. Values must be integers between 1 and 5.
- Do not rename, remove, reorder, or add columns.
- Save as `.xlsx` or `.csv`.

## Benchmarking Template
Required columns (must match exactly):
{BENCH_REQUIRED}

- `MarketMedianCTC` should be numeric (annual INR).
- Save as `.xlsx` or `.csv`.

## Upload rules
1. Download the two official templates and this guide.
2. Populate only the columns provided; do not add extra sheets in Excel files.
3. Return to this page and confirm you downloaded the templates, then upload your files.
4. If headers do not match exactly the upload will be blocked.

**PerformanceRating reminder:** This dashboard interprets **1 as highest** and **5 as lowest**. Please prepare your data accordingly.
"""
    return md

def create_howto_pdf_bytes():
    """Small PDF version of the how-to guide (ReportLab)"""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []
    # Title
    title_style = ParagraphStyle("title", parent=styles["Title"], fontName=HEADER_FONT, fontSize=20, alignment=1, textColor=TEXT_COLOR)
    story.append(Paragraph("How to Upload Data — C&B Dashboard", title_style))
    story.append(Spacer(1, 8))
    body_style = ParagraphStyle("body", parent=styles["Normal"], fontName=BODY_FONT, fontSize=10, leading=14, textColor=TEXT_COLOR)
    # Add content lines
    md = get_howto_markdown()
    for para in md.split("\n\n"):
        story.append(Paragraph(para.replace("\n","<br/>"), body_style))
        story.append(Spacer(1,6))
    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# -----------------------
# Sidebar: Template download and precondition
# -----------------------
st.sidebar.header("Step 1: Templates & Guide")
st.sidebar.write("Download the official templates and the how-to guide. You must confirm you have downloaded before uploading data.")

colA, colB = st.sidebar.columns(2)
with colA:
    st.sidebar.download_button("📥 Employee Template (.csv)", data=get_employee_template_csv(), file_name="Employee_Template.csv", mime="text/csv")
with colB:
    st.sidebar.download_button("📥 Benchmark Template (.csv)", data=get_benchmark_template_csv(), file_name="Benchmark_Template.csv", mime="text/csv")

st.sidebar.download_button("📄 How-to Upload Guide (PDF)", data=create_howto_pdf_bytes(), file_name="How_to_Upload_Guide.pdf", mime="application/pdf")

confirm_download = st.sidebar.checkbox("✅ I downloaded the templates and read the How-to guide (required)", value=False)

if not confirm_download:
    st.sidebar.info("Please download templates & guide before proceeding.")
    st.stop()

# -----------------------
# Upload Section - enforce download confirmation
# -----------------------
st.header("Upload Data (strict templates required)")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📂 Upload Employee Compensation file (.csv / .xlsx)", type=["csv","xlsx"])
with col2:
    benchmark_file = st.file_uploader("📂 Upload Benchmarking file (.csv / .xlsx) [optional]", type=["csv","xlsx"])

if not uploaded_file:
    st.info("Please upload the Employee Compensation file to start.")
    st.stop()

# -----------------------
# Load and validate uploaded data
# -----------------------
def read_file(file):
    if file is None:
        return None
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, engine="openpyxl")
    return df

emp_df = read_file(uploaded_file)

# make sure columns haven't been auto-normalized; we require exact headers for validation first
# But we'll also accept .csv's where Excel might have trimmed whitespace; to be strict, require exact headers.
if emp_df is None:
    st.error("Unable to read uploaded employee file.")
    st.stop()

# If columns are not exact, show the preview and fail
ok, msg = validate_exact_headers(list(emp_df.columns), EMP_REQUIRED), None
if not ok:
    # Try to repair common issues where headers may have extra spaces or lower/upper case by stripping whitespace & exact map
    # But per requirement: strict exact matching. So show explicit error and fail.
    st.error("Employee file headers do not exactly match required template.")
    st.write("Expected:", EMP_REQUIRED)
    st.write("Found:", list(emp_df.columns))
    st.stop()

# Convert numeric columns
emp_df["CTC"] = pd.to_numeric(emp_df["CTC"], errors="coerce")
emp_df["Bonus"] = pd.to_numeric(emp_df["Bonus"], errors="coerce")
emp_df["PerformanceRating"] = pd.to_numeric(emp_df["PerformanceRating"], errors="coerce")

# Validate performance rating constraints
if not emp_df["PerformanceRating"].dropna().between(1,5).all():
    st.error("PerformanceRating must be integers between 1 and 5 (1 = highest). Please fix and re-upload.")
    st.stop()

# Optional benchmark
bench_df = None
if benchmark_file:
    bench_df = read_file(benchmark_file)
    if bench_df is None:
        st.warning("Could not read benchmark file. Ignoring benchmark.")
        bench_df = None
    else:
        ok_b, _ = validate_exact_headers(list(bench_df.columns), BENCH_REQUIRED), None
        if not ok_b:
            st.error("Benchmark file headers do not exactly match required template.")
            st.write("Expected:", BENCH_REQUIRED)
            st.write("Found:", list(bench_df.columns))
            st.stop()
        bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce")

# -----------------------
# Preview uploaded data (StepB.1 kept as-is)
# -----------------------
st.subheader("👀 Preview Uploaded Data")
st.write("Showing first 10 rows of uploaded employee file (exact headers enforced).")
st.dataframe(emp_df.head(10), use_container_width=True)

if bench_df is not None:
    st.write("Benchmark file preview:")
    st.dataframe(bench_df.head(10), use_container_width=True)

# -----------------------
# Filter controls (dependent: Department -> JobRole) (global)
# -----------------------
st.sidebar.header("Filters (Dept → JobRole)")
departments = ["All"] + sorted(emp_df["Department"].dropna().unique().tolist())
selected_dept = st.sidebar.selectbox("Department", departments, index=0)

available_roles = []
if selected_dept != "All":
    available_roles = sorted(emp_df.loc[emp_df["Department"] == selected_dept, "JobRole"].dropna().unique().tolist())
else:
    available_roles = sorted(emp_df["JobRole"].dropna().unique().tolist())

# If there's only one role for the department, show it but as a non-mandatory multiselect (auto-selected)
if len(available_roles) == 0:
    selected_roles = []
    st.sidebar.warning("No job roles available for selected department.")
else:
    if len(available_roles) == 1:
        selected_roles = st.sidebar.multiselect("Job Role (dependent)", available_roles, default=available_roles)
    else:
        selected_roles = st.sidebar.multiselect("Job Role (dependent)", available_roles, default=[])

def apply_filters(df, dept, roles):
    d = df.copy()
    if dept != "All":
        d = d[d["Department"] == dept]
    if roles:
        d = d[d["JobRole"].isin(roles)]
    return d

filtered_df = apply_filters(emp_df, selected_dept, selected_roles)

if filtered_df.shape[0] == 0:
    st.warning("No data after applying filters. Broaden your selection.")
    st.stop()

# -----------------------
# Utility: Quartile binning function (per StepB.6)
# -----------------------
def make_quartile_categorizer(series):
    """Return categorizer function and quantiles/fences used."""
    q1, q2, q3 = series.quantile([0.25, 0.5, 0.75]).tolist()
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    def categorize(x):
        if pd.isnull(x):
            return "NA"
        if x < lower_fence:
            return "Below Min"
        if x <= q1:
            return "Q1"
        if x <= q2:
            return "Q2"
        if x <= q3:
            return "Q3"
        if x <= upper_fence:
            return "Q4"
        return "Beyond Max"
    meta = {"q1": q1, "q2": q2, "q3": q3, "lower_fence": lower_fence, "upper_fence": upper_fence}
    return categorize, meta

# -----------------------
# Helpers: Build metric PNG + per-metric PDF section (ReportLab)
# -----------------------
def plot_and_save_bar(df_plot, x, y, title, filename_prefix):
    fig = px.bar(df_plot, x=x, y=y, color=x, title=title)
    p = save_plotly_png(fig, f"{filename_prefix}.png")
    return fig, p

def create_metric_pdf_section(story, title, description, table_df=None, chart_path=None, styles=None):
    # title
    story.append(Paragraph(title, styles["Heading2"]))
    if description:
        story.append(Paragraph(description, styles["BodyText"]))
        story.append(Spacer(1,6))
    # table
    if table_df is not None:
        data = [list(table_df.columns)] + table_df.head(200).fillna("").values.tolist()
        tstyle = TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black), ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)])
        # zebra
        for i in range(1, len(data)):
            if i % 2 == 0:
                tstyle.add("BACKGROUND", (0,i), (-1,i), TABLE_ZEBRA)
        tbl = Table(data, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(tstyle)
        story.append(tbl)
        story.append(Spacer(1,6))
    # chart image
    if chart_path and os.path.exists(chart_path):
        story.append(RLImage(chart_path, width=170*mm, height=90*mm))
        story.append(Spacer(1,6))
    story.append(PageBreak())

# -----------------------
# METRICS (StepB list implemented)
# Each metric area should include:
# - table view
# - chart view (plotly visible)
# - export image (PNG) for the chart
# - export per-metric PDF
# - add to sections/images/tables list for master compile
# -----------------------
sections = []   # tuples (title, description)
images = []     # tuples (path, caption)
tables = []     # tuples (title, dataframe)

# ---------- Metric A: Average CTC by JobLevel (StepB.2)
st.subheader("🏷️ Average CTC by Job Level")
avg_ctc_joblevel = filtered_df.groupby("JobLevel", sort=False)["CTC"].mean().reset_index()
avg_ctc_joblevel["CTC_readable"] = avg_ctc_joblevel["CTC"].apply(readable_currency)
st.dataframe(avg_ctc_joblevel[["JobLevel","CTC_readable"]].rename(columns={"CTC_readable":"AvgCTC"}), use_container_width=True)

fig_avg_jl = px.bar(avg_ctc_joblevel, x="JobLevel", y="CTC", title="Average CTC by Job Level")
png_path_avg_jl = save_plotly_png(fig_avg_jl, safe_filename("avg_ctc_joblevel"))
st.plotly_chart(fig_avg_jl, use_container_width=True)

# per-metric exports
col_ea, col_eb = st.columns([1,1])
with col_ea:
    img_bytes = open(png_path_avg_jl, "rb").read()
    st.download_button("📸 Export Image (PNG)", data=img_bytes, file_name=os.path.basename(png_path_avg_jl), mime="image/png")
with col_eb:
    if st.button("📄 Export Metric PDF - Average CTC by Job Level"):
        # build single metric pdf bytes and provide download
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        story = []
        create_metric_pdf_section(story, "Average CTC by Job Level", "Average CTC across Job Levels (filtered selection).", avg_ctc_joblevel, png_path_avg_jl, styles)
        doc.build(story)
        pdf_bytes = buf.getvalue()
        buf.close()
        st.download_button("⬇️ Download PDF (Avg CTC by Job Level)", data=pdf_bytes, file_name="avg_ctc_joblevel.pdf", mime="application/pdf")

sections.append(("Average CTC by Job Level", "Shows average compensation across job levels."))
images.append((png_path_avg_jl, "Average CTC by Job Level"))
tables.append(("Average CTC by Job Level", avg_ctc_joblevel))

# ---------- Metric B: Median CTC by JobLevel (replaces Avg by JobRole) (StepB.4)
st.subheader("📏 Median CTC by Job Level")
median_ctc_joblevel = filtered_df.groupby("JobLevel", sort=False)["CTC"].median().reset_index().rename(columns={"CTC":"MedianCTC"})
median_ctc_joblevel["MedianCTC_readable"] = median_ctc_joblevel["MedianCTC"].apply(readable_currency)
st.dataframe(median_ctc_joblevel[["JobLevel","MedianCTC_readable"]].rename(columns={"MedianCTC_readable":"MedianCTC"}), use_container_width=True)

fig_med_jl = px.bar(median_ctc_joblevel, x="JobLevel", y="MedianCTC", color="JobLevel", title="Median CTC by Job Level")
png_path_med_jl = save_plotly_png(fig_med_jl, safe_filename("median_ctc_joblevel"))
st.plotly_chart(fig_med_jl, use_container_width=True)

col_ba, col_bb = st.columns([1,1])
with col_ba:
    st.download_button("📸 Export Image (PNG) - Median CTC", data=open(png_path_med_jl,"rb").read(), file_name=os.path.basename(png_path_med_jl), mime="image/png")
with col_bb:
    if st.button("📄 Export Metric PDF - Median CTC by Job Level"):
        buf = BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm); styles = getSampleStyleSheet(); story=[]
        create_metric_pdf_section(story, "Median CTC by Job Level", "Median compensation across job levels (filtered selection).", median_ctc_joblevel, png_path_med_jl, styles)
        doc.build(story); pdf_bytes = buf.getvalue(); buf.close()
        st.download_button("⬇️ Download PDF (Median CTC by Job Level)", data=pdf_bytes, file_name="median_ctc_joblevel.pdf", mime="application/pdf")

sections.append(("Median CTC by Job Level", "Shows median compensation across job levels."))
images.append((png_path_med_jl, "Median CTC by Job Level"))
tables.append(("Median CTC by Job Level", median_ctc_joblevel))

# ---------- Metric C: Quartile Placement by JobLevel (StepB.6) - aggregated table
st.subheader("📊 Quartile Placement by Job Level")

def build_quartile_table(df, group_col="JobLevel"):
    rows=[]
    for lvl, g in df.groupby(group_col, sort=False):
        cat_func, meta = make_quartile_categorizer(g["CTC"])
        cats = g["CTC"].apply(cat_func)
        vc = cats.value_counts()
        row = {
            "JobLevel": lvl,
            "Count": len(g),
            "Below Min": int(vc.get("Below Min", 0)),
            "Q1": int(vc.get("Q1", 0)),
            "Q2": int(vc.get("Q2", 0)),
            "Q3": int(vc.get("Q3", 0)),
            "Q4": int(vc.get("Q4", 0)),
            "Beyond Max": int(vc.get("Beyond Max", 0))
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    # totals row
    if not out.empty:
        totals = out[["Count","Below Min","Q1","Q2","Q3","Q4","Beyond Max"]].sum()
        total_row = {"JobLevel":"Total"}
        total_row.update(totals.to_dict())
        grand = totals["Count"]
        perc_row = {"JobLevel":"Total (%)"}
        for c in ["Below Min","Q1","Q2","Q3","Q4","Beyond Max"]:
            perc_row[c] = round(100.0 * totals[c] / grand, 2) if grand>0 else 0.0
        out = pd.concat([out, pd.DataFrame([total_row, perc_row])], ignore_index=True)
    return out

quartile_table = build_quartile_table(filtered_df, group_col="JobLevel")
st.dataframe(quartile_table, use_container_width=True)

# boxplot (quartile distribution by JobLevel) - StepB.5 / B.11 visuals
fig_box = px.box(filtered_df, x="JobLevel", y="CTC", points="all", title="Quartile Distribution (by JobLevel)")
png_path_box = save_plotly_png(fig_box, safe_filename("quartile_boxplot"))
st.plotly_chart(fig_box, use_container_width=True)

col_qa, col_qb = st.columns([1,1])
with col_qa:
    st.download_button("📸 Export Image (PNG) - Quartile", data=open(png_path_box,"rb").read(), file_name=os.path.basename(png_path_box), mime="image/png")
with col_qb:
    if st.button("📄 Export Metric PDF - Quartile Placement"):
        buf = BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm); styles=getSampleStyleSheet(); story=[]
        create_metric_pdf_section(story, "Quartile Placement by Job Level", "Quartile placement table and distribution (company data).", quartile_table, png_path_box, styles)
        doc.build(story); pdf_bytes = buf.getvalue(); buf.close()
        st.download_button("⬇️ Download PDF (Quartile Placement)", data=pdf_bytes, file_name="quartile_placement.pdf", mime="application/pdf")

sections.append(("Quartile Placement", "Shows distribution buckets (Below Min, Q1..Q4, Beyond Max) for each Job Level."))
images.append((png_path_box, "Quartile Distribution"))
tables.append(("Quartile Placement", quartile_table))

# ---------- Metric D: Bonus % by JobLevel (StepB.8)
st.subheader("🎁 Average Bonus % of CTC by Job Level")
filtered_df["BonusPct"] = np.where(filtered_df["CTC"]>0, (filtered_df["Bonus"]/filtered_df["CTC"])*100.0, np.nan)
bonus_by_level = filtered_df.groupby("JobLevel", sort=False)["BonusPct"].mean().reset_index()
bonus_by_level["BonusPct"] = bonus_by_level["BonusPct"].round(2)
st.dataframe(bonus_by_level, use_container_width=True)

fig_bonus = px.bar(bonus_by_level, x="JobLevel", y="BonusPct", color="JobLevel", title="Avg Bonus % by JobLevel")
png_path_bonus = save_plotly_png(fig_bonus, safe_filename("bonus_pct"))
st.plotly_chart(fig_bonus, use_container_width=True)

col_ba1, col_ba2 = st.columns([1,1])
with col_ba1:
    st.download_button("📸 Export Image (PNG) - Bonus %", data=open(png_path_bonus,"rb").read(), file_name=os.path.basename(png_path_bonus), mime="image/png")
with col_ba2:
    if st.button("📄 Export Metric PDF - Bonus % by Job Level"):
        buf = BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm); styles=getSampleStyleSheet(); story=[]
        create_metric_pdf_section(story, "Average Bonus % of CTC by Job Level", "Average bonus percentage across job levels.", bonus_by_level, png_path_bonus, styles)
        doc.build(story); pdf_bytes = buf.getvalue(); buf.close()
        st.download_button("⬇️ Download PDF (Bonus % by Job Level)", data=pdf_bytes, file_name="bonus_pct_joblevel.pdf", mime="application/pdf")

sections.append(("Avg Bonus % of CTC", "Shows average bonus percent across job levels."))
images.append((png_path_bonus, "Avg Bonus % of CTC"))
tables.append(("Avg Bonus % of CTC", bonus_by_level))

# ---------- Metric E: Company vs Market Benchmarking (StepB.10 & StepB.11)
if bench_df is not None:
    st.subheader("📉 Company vs Market Benchmarking (Median CTC)")
    # compute company median by JobLevel (not JobRole as per revised spec)
    comp_med_level = filtered_df.groupby("JobLevel", sort=False)["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
    bench_med_level = bench_df.groupby("JobLevel", sort=False)["MarketMedianCTC"].median().reset_index().rename(columns={"MarketMedianCTC":"MarketMedian"})
    compare = pd.merge(comp_med_level, bench_med_level, on="JobLevel", how="outer").fillna(0)
    compare["MedianGapPct"] = np.where(compare["MarketMedian"]>0, (compare["CompanyMedian"] - compare["MarketMedian"]) / compare["MarketMedian"] * 100.0, np.nan)
    compare["MedianGapPct"] = compare["MedianGapPct"].round(2)
    st.dataframe(compare, use_container_width=True)

    # Bar (Company median) + line (Market median)
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(x=compare["JobLevel"], y=compare["CompanyMedian"], name="CompanyMedian"))
    fig_cmp.add_trace(go.Scatter(x=compare["JobLevel"], y=compare["MarketMedian"], name="MarketMedian", mode="lines+markers", line=dict(width=3)))
    fig_cmp.update_layout(title="Company Median (bars) vs Market Median (line)", xaxis_title="JobLevel", yaxis_title="CTC")
    png_path_cmp = save_plotly_png(fig_cmp, safe_filename("company_vs_market"))
    st.plotly_chart(fig_cmp, use_container_width=True)

    col_cmp1, col_cmp2 = st.columns([1,1])
    with col_cmp1:
        st.download_button("📸 Export Image (PNG) - Company vs Market", data=open(png_path_cmp,"rb").read(), file_name=os.path.basename(png_path_cmp), mime="image/png")
    with col_cmp2:
        if st.button("📄 Export Metric PDF - Company vs Market"):
            # build PDF with conditional color commentary below the table
            buf = BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm); styles=getSampleStyleSheet(); story=[]
            # table + chart
            create_metric_pdf_section(story, "Company vs Market Benchmarking (Median CTC)", "Company vs Market medians by job level. Median gap % column uses conditional coloring internally.", compare, png_path_cmp, styles)
            # Add remarks summary (top 3 behind and top 3 ahead)
            story.append(Paragraph("Remarks & Insights", styles["Heading3"]))
            # compute top negative and top positive
            tmp = compare.dropna(subset=["MedianGapPct"])
            top_neg = tmp.nsmallest(3, "MedianGapPct")
            top_pos = tmp.nlargest(3, "MedianGapPct")
            for _, r in top_neg.iterrows():
                story.append(Paragraph(f"⚠️ {r['JobLevel']} is {r['MedianGapPct']}% behind market median. Consider review.", styles["Normal"]))
            for _, r in top_pos.iterrows():
                story.append(Paragraph(f"✅ {r['JobLevel']} is {r['MedianGapPct']}% ahead of market median. Competitive placement.", styles["Normal"]))
            story.append(PageBreak())
            doc.build(story)
            pdf_bytes = buf.getvalue(); buf.close()
            st.download_button("⬇️ Download PDF (Company vs Market)", data=pdf_bytes, file_name="company_vs_market.pdf", mime="application/pdf")

    sections.append(("Company vs Market (Median)", "Comparison of company vs market median CTC (per JobLevel)."))
    images.append((png_path_cmp, "Company vs Market Median"))
    tables.append(("Company vs Market", compare))

# -----------------------
# Downloads area (StepB.12)
# - Checkbox grid listing KPI blocks (we will use the sections list)
# - Compile selected KPI PDFs into a single PDF with cover + TOC + sections + consolidated conclusion
# -----------------------
st.header("📥 Download Reports & Images")

# Build KPI grid from sections (sections list built earlier)
kpi_titles = [s[0] for s in sections]
kpi_check = {}
st.write("Select KPIs to include in compiled report:")
cols = st.columns(3)
for i, title in enumerate(kpi_titles):
    with cols[i%3]:
        kpi_check[title] = st.checkbox(title, value=False)

if st.button("🧩 Compile Selected KPIs into Single PDF"):
    selected = [k for k,v in kpi_check.items() if v]
    if not selected:
        st.warning("Select at least one KPI to compile.")
    else:
        # Build master PDF: cover -> TOC -> sections for selected KPIs -> consolidated insights
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        story = []
        # Cover
        cover_style = ParagraphStyle("Cover", parent=styles["Title"], fontName=HEADER_FONT, fontSize=22, alignment=1, textColor=TEXT_COLOR)
        story.append(Paragraph("Compensation & Benefits - Compiled Report", cover_style))
        story.append(Spacer(1,6))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}", styles["Normal"]))
        story.append(PageBreak())
        # TOC
        story.append(Paragraph("Table of Contents", styles["Heading2"]))
        for idx, t in enumerate(selected,1):
            story.append(Paragraph(f'<a href="#sec{idx}">{idx}. {t}</a>', styles["Normal"]))
        story.append(PageBreak())
        # Sections (pull from tables/images)
        for idx, title in enumerate(selected,1):
            # find table and image for that title
            story.append(Paragraph(f'<a name="sec{idx}"/>{title}', styles["Heading2"]))
            # find table in tables list
            table_df = None
            for ttitle, df in tables:
                if ttitle == title:
                    table_df = df
                    break
            # find image
            image_path = None
            for ipath, caption in images:
                if caption == title:
                    image_path = ipath
                    break
            # create section
            create_metric_pdf_section(story, title, f"Auto-generated section for {title}.", table_df, image_path, styles)
        # Consolidated Conclusion: auto-generate basic bullets
        story.append(Paragraph("Consolidated Conclusions", styles["Heading2"]))
        # Very simple synthesis: list top 3 quartile risk levels and top gaps if available
        if "compare" in locals():
            tmp = compare.dropna(subset=["MedianGapPct"]).sort_values("MedianGapPct")
            # worst 3
            worst = tmp.head(3)
            best = tmp.tail(3)
            for _, r in worst.iterrows():
                story.append(Paragraph(f"⚠️ {r['JobLevel']} behind market by {r['MedianGapPct']}% — consider calibration.", styles["Normal"]))
            for _, r in best.iterrows():
                story.append(Paragraph(f"✅ {r['JobLevel']} ahead of market by {r['MedianGapPct']}% — placement competitive.", styles["Normal"]))
        else:
            story.append(Paragraph("No benchmark data supplied to derive market conclusions.", styles["Normal"]))
        # finalize
        doc.build(story)
        pdf_bytes = buf.getvalue(); buf.close()
        st.download_button("⬇️ Download Compiled PDF", data=pdf_bytes, file_name="cb_compiled_report.pdf", mime="application/pdf")

# Also show direct image downloads (clean downloads section)
st.markdown("### Quick image downloads for slide/ppt")
imgs_map = {caption:path for path,caption in images}
for caption, path in imgs_map.items():
    if os.path.exists(path):
        with open(path,"rb") as f:
            st.download_button(f"⬇️ {caption}", data=f.read(), file_name=os.path.basename(path), mime="image/png")

# End of app
st.success("Dashboard loaded. Use filters on the left to drive all metrics. Export PNG / per-metric PDF using the buttons next to each metric. Use the compiled report generator to produce a multi-KPI PDF.")