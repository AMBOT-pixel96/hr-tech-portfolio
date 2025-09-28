# ============================================================
# cb_dashboard.py — Compensation & Benefits Dashboard
# Version: 2.8 (Kaleido-safe exports + Full app)
# Last Updated: 2025-09-29 19:15 IST
# Notes:
# - Step A + Step B fully implemented
# - App header always visible (banner + shields.io badges)
# - Step 1: Templates + Guide, Step 2: Upload, Step 3: Filters
# - Metrics A–E implemented with PNG + PDF exports
# - Plotly PNG export attempts kaleido and falls back to HTML without crashing the app
# - Compiled PDF: Cover + TOC + tables + charts + insights
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# import seaborn as sns   # kept for reference if visually needed later
from io import BytesIO
from datetime import datetime
import os
import math

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

# -----------------------
# Low-level PDF page utilities
# -----------------------
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
def safe_filename(prefix):
    # return prefix without extension — timestamp appended
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def readable_currency(x):
    if pd.isna(x):
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 1e6:
        return f"₹{x/1e6:,.2f}M"
    return f"₹{x:,.0f}"

def sanitize_anchor(title: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in title).strip("_")

def validate_exact_headers(df_or_cols, required_cols):
    """Accept DataFrame or list-like of columns."""
    if hasattr(df_or_cols, "columns"):
        cols = list(df_or_cols.columns)
    else:
        cols = list(df_or_cols)
    if cols == required_cols:
        return True, "OK"
    return False, f"Header mismatch. Expected {required_cols}, found {cols}"

def save_plotly_asset(fig, filename_base, width=1200, height=700, scale=2):
    """
    Attempt to save a Plotly figure as PNG using kaleido.
    On failure, write HTML fallback.
    Returns dict: {"png": path_or_None, "html": path_or_None, "error": error_str_or_None}
    """
    base = os.path.join(TMP_DIR, filename_base)
    png_path = base + ".png"
    html_path = base + ".html"
    try:
        # primary attempt (requires kaleido installed)
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        with open(png_path, "wb") as f:
            f.write(img_bytes)
        return {"png": png_path, "html": None, "error": None}
    except Exception as e_png:
        try:
            # fallback: export an interactive HTML snapshot
            fig.write_html(html_path)
            return {"png": None, "html": html_path, "error": str(e_png)}
        except Exception as e_html:
            return {"png": None, "html": None, "error": f"png_err:{e_png} | html_err:{e_html}"}

# Small wrapper to choose correct download mime and file content
def read_binary(path):
    mode = "rb"
    with open(path, mode) as f:
        return f.read()
# -----------------------
# Templates & How-to Guide
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
# Welcome Banner + Badges (neutral background — not pink)
# -----------------------
st.markdown(
    f"""
    <div style="padding:20px;border-radius:10px;
                border:1px solid rgba(255,255,255,0.06);text-align:center">
        <h1 style="margin-bottom:0;color:inherit">📊 Compensation & Benefits Dashboard</h1>
        <p style="margin-top:5px;font-size:14px;color:inherit;">
            Analyze pay structures, benchmark against market, and export
            boardroom-ready reports with one click.
        </p>
        <div style="display:flex;justify-content:center;gap:8px;margin-top:10px;">
            <img src="https://img.shields.io/badge/version-2.8-blue?style=flat-square"/>
            <img src="https://img.shields.io/badge/streamlit-cloud-red?style=flat-square&logo=streamlit"/>
            <img src="https://img.shields.io/badge/python-3.10+-yellow?style=flat-square&logo=python"/>
            <img src="https://img.shields.io/badge/reportlab-PDF-green?style=flat-square"/>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Step 1: Templates & Guide (main page)
# -----------------------
st.header("Step 1: Download Templates & Guide")
colA, colB = st.columns(2)
with colA:
    st.download_button("📥 Internal Compensation Data Template",
        data=get_employee_template_csv(),
        file_name="Internal_Compensation_Data_Template.csv", mime="text/csv")
with colB:
    st.download_button("📥 External Benchmarking Data Template",
        data=get_benchmark_template_csv(),
        file_name="External_Benchmarking_Data_Template.csv", mime="text/csv")

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
    uploaded_file = st.file_uploader("📂 Upload Internal Compensation CSV/XLSX",
                                     type=["csv","xlsx"])
with col2:
    benchmark_file = st.file_uploader("📂 Upload External Benchmarking CSV/XLSX [optional]",
                                      type=["csv","xlsx"])

if not uploaded_file:
    st.warning("Please upload the Internal Compensation file.")
    st.stop()

def read_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")

emp_df = read_file(uploaded_file)
ok, msg = validate_exact_headers(emp_df, EMP_REQUIRED)
if not ok:
    st.error(msg); st.stop()

# Ensure numeric columns
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
# Step 3: Filters (kept on main-screen layout but using columns)
# -----------------------
st.header("Step 3: Filters")
fcol1, fcol2 = st.columns([2,4])
with fcol1:
    departments = ["All"] + sorted(emp_df["Department"].dropna().unique())
    selected_dept = st.selectbox("Department", departments, index=0)
    if selected_dept != "All":
        roles = sorted(emp_df.loc[emp_df["Department"]==selected_dept,"JobRole"].dropna().unique())
    else:
        roles = sorted(emp_df["JobRole"].dropna().unique())
    if len(roles) == 0:
        selected_roles = []
        st.warning("No job roles for selected dept.")
    elif len(roles) == 1:
        selected_roles = st.multiselect("Job Role", roles, default=roles)
    else:
        selected_roles = st.multiselect("Job Role", roles, default=[])

def apply_filters(df, dept, roles):
    d = df.copy()
    if dept != "All":
        d = d[d["Department"]==dept]
    if roles:
        d = d[d["JobRole"].isin(roles)]
    return d

filtered_df = apply_filters(emp_df, selected_dept, selected_roles)
if filtered_df.shape[0] == 0:
    st.warning("No data after filters."); st.stop()

# -----------------------
# Quartile categorizer
# -----------------------
def make_quartile_categorizer(series):
    q1, q2, q3 = series.quantile([0.25,0.5,0.75]).tolist()
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
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
# Per-metric PDF generator (supports chart info dict)
# -----------------------
def create_metric_pdf_bytes(title, description, table_df=None, chart_info=None):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        rightMargin=18*mm, leftMargin=18*mm,
        topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Spacer(1,6))
    if description:
        story.append(Paragraph(description, styles["Normal"]))
        story.append(Spacer(1,6))
    if table_df is not None:
        data = [list(table_df.columns)] + table_df.fillna("").values.tolist()
        tstyle = TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),
                             ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)])
        for i in range(1,len(data)):
            if i%2==0:
                tstyle.add("BACKGROUND",(0,i),(-1,i),TABLE_ZEBRA)
        tbl = Table(data, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(tstyle)
        story.append(tbl)
        story.append(Spacer(1,6))
    if chart_info:
        # chart_info is dict {"png":..., "html":..., "error":...}
        if chart_info.get("png") and os.path.exists(chart_info["png"]):
            story.append(RLImage(chart_info["png"], width=170*mm, height=90*mm))
            story.append(Spacer(1,6))
        elif chart_info.get("html") and os.path.exists(chart_info["html"]):
            story.append(Paragraph(f"Interactive chart saved as HTML: {os.path.basename(chart_info['html'])}", styles["Normal"]))
            story.append(Spacer(1,6))
            if chart_info.get("error"):
                story.append(Paragraph(f"Note: PNG export failed with error: {chart_info['error']}", styles["Normal"]))
                story.append(Spacer(1,6))
    doc.build(story,
              onFirstPage=lambda c,d:(draw_background(c,d), add_page_number(c,d)),
              onLaterPages=lambda c,d:(draw_background(c,d), add_page_number(c,d)))
    return buf.getvalue()

# -----------------------
# Metrics A–E implementation with safe asset saving
# -----------------------
sections, images, tables = [], [], []

# Metric A: Average CTC by Job Level
st.subheader("🏷️ Average CTC by Job Level")
avg_ctc_jl = filtered_df.groupby("JobLevel", sort=False)["CTC"].mean().reset_index()
avg_ctc_jl["CTC_fmt"] = avg_ctc_jl["CTC"].apply(readable_currency)
st.dataframe(avg_ctc_jl[["JobLevel","CTC_fmt"]], use_container_width=True)
fig_avg = px.bar(avg_ctc_jl, x="JobLevel", y="CTC", title="Average CTC by Job Level")
asset_avg = save_plotly_asset(fig_avg, safe_filename("avg_ctc_joblevel"))
st.plotly_chart(fig_avg, use_container_width=True)
sections.append(("Average CTC by Job Level","Shows average pay across job levels."))
images.append((asset_avg, "Average CTC by Job Level"))
tables.append(("Average CTC by Job Level", avg_ctc_jl))

colA, colB = st.columns(2)
with colA:
    if asset_avg.get("png") and os.path.exists(asset_avg["png"]):
        with open(asset_avg["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG)", f.read(), file_name=os.path.basename(asset_avg["png"]), mime="image/png")
    elif asset_avg.get("html") and os.path.exists(asset_avg["html"]):
        with open(asset_avg["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML)", f.read(), file_name=os.path.basename(asset_avg["html"]), mime="text/html")
    else:
        st.info("No chart asset available for download.")
with colB:
    st.download_button("⬇️ Export Descriptive PDF",
                       create_metric_pdf_bytes("Average CTC by Job Level",
                           "Shows average pay across job levels. Interpretation: highest levels show ... (customize as needed).",
                           avg_ctc_jl, asset_avg),
                       file_name="avg_ctc_joblevel.pdf", mime="application/pdf")

# Metric B: Median CTC by Job Level
st.subheader("📏 Median CTC by Job Level")
med_ctc_jl = filtered_df.groupby("JobLevel", sort=False)["CTC"].median().reset_index().rename(columns={"CTC":"MedianCTC"})
med_ctc_jl["MedianCTC_fmt"] = med_ctc_jl["MedianCTC"].apply(readable_currency)
st.dataframe(med_ctc_jl[["JobLevel","MedianCTC_fmt"]], use_container_width=True)
fig_med = px.bar(med_ctc_jl, x="JobLevel", y="MedianCTC", title="Median CTC by Job Level")
asset_med = save_plotly_asset(fig_med, safe_filename("median_ctc_joblevel"))
st.plotly_chart(fig_med, use_container_width=True)
sections.append(("Median CTC by Job Level","Shows median pay across job levels."))
images.append((asset_med,"Median CTC by Job Level"))
tables.append(("Median CTC by Job Level",med_ctc_jl))

colA, colB = st.columns(2)
with colA:
    if asset_med.get("png") and os.path.exists(asset_med["png"]):
        with open(asset_med["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG)", f.read(), file_name=os.path.basename(asset_med["png"]), mime="image/png")
    elif asset_med.get("html") and os.path.exists(asset_med["html"]):
        with open(asset_med["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML)", f.read(), file_name=os.path.basename(asset_med["html"]), mime="text/html")
    else:
        st.info("No chart asset available for download.")
with colB:
    st.download_button("⬇️ Export Descriptive PDF",
                       create_metric_pdf_bytes("Median CTC by Job Level",
                           "Shows median pay across job levels. Useful to identify central tendency by level.",
                           med_ctc_jl, asset_med),
                       file_name="median_ctc_joblevel.pdf", mime="application/pdf")

# Metric C: Quartile Placement by Job Level
st.subheader("📊 Quartile Placement by Job Level")
def build_quartile_table(df, group_col="JobLevel"):
    rows = []
    for lvl, g in df.groupby(group_col, sort=False):
        cat_func, _ = make_quartile_categorizer(g["CTC"])
        cats = g["CTC"].apply(cat_func)
        vc = cats.value_counts()
        row = {"JobLevel": lvl, "Count": len(g),
               "Q1": int(vc.get("Q1",0)), "Q2": int(vc.get("Q2",0)),
               "Q3": int(vc.get("Q3",0)), "Q4": int(vc.get("Q4",0)),
               "Outlier": int(vc.get("Outlier",0))}
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        totals = out.drop(columns=["JobLevel"]).sum()
        out.loc[len(out)] = {"JobLevel":"Total", **totals.to_dict()}
        grand = totals["Count"] if "Count" in totals else 0
        perc = {c: round(100*totals[c]/grand,2) if grand>0 else 0 for c in ["Q1","Q2","Q3","Q4","Outlier"]}
        perc.update({"Count":100})
        out.loc[len(out)] = {"JobLevel":"Total (%)", **perc}
    return out

quartile_tbl = build_quartile_table(filtered_df)
st.dataframe(quartile_tbl, use_container_width=True)
fig_box = px.box(filtered_df, x="JobLevel", y="CTC", points="all", title="CTC Distribution by JobLevel")
asset_box = save_plotly_asset(fig_box, safe_filename("quartile_box"))
st.plotly_chart(fig_box, use_container_width=True)
sections.append(("Quartile Placement","Distribution by quartiles + outliers per Job Level."))
images.append((asset_box,"Quartile Placement"))
tables.append(("Quartile Placement", quartile_tbl))

colA, colB = st.columns(2)
with colA:
    if asset_box.get("png") and os.path.exists(asset_box["png"]):
        with open(asset_box["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG)", f.read(), file_name=os.path.basename(asset_box["png"]), mime="image/png")
    elif asset_box.get("html") and os.path.exists(asset_box["html"]):
        with open(asset_box["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML)", f.read(), file_name=os.path.basename(asset_box["html"]), mime="text/html")
    else:
        st.info("No chart asset available for download.")
with colB:
    st.download_button("⬇️ Export Descriptive PDF",
                       create_metric_pdf_bytes("Quartile Placement by Job Level",
                           "Quartile placement and counts across job levels. Highlights outliers and distribution skew.",
                           quartile_tbl, asset_box),
                       file_name="quartile_placement.pdf", mime="application/pdf")

# Metric C2: Quadrant Concentration (scatter)
st.subheader("⚫ Compensation Quadrant Concentration")
quad_df = filtered_df.copy()
cat_func, _ = make_quartile_categorizer(quad_df["CTC"])
quad_df["QuartileCat"] = quad_df["CTC"].apply(cat_func)
quad_df["QuadBucket"] = quad_df["QuartileCat"].apply(lambda c: c if c in ["Q1","Q2","Q3","Q4"] else "Outlier")
quad_df["PctWithinLevel"] = quad_df.groupby("JobLevel")["CTC"].rank(pct=True) * 100.0
fig_quad = px.scatter(quad_df, x="PctWithinLevel", y="CTC", color="QuadBucket",
                      hover_data=["EmployeeID","Department","JobRole","JobLevel"],
                      title="Quadrant Concentration by Percentile & CTC")
fig_quad.add_vline(x=25, line_dash="dash", line_color="lightgrey")
fig_quad.add_vline(x=50, line_dash="dash", line_color="lightgrey")
fig_quad.add_vline(x=75, line_dash="dash", line_color="lightgrey")
asset_quad = save_plotly_asset(fig_quad, safe_filename("quadrant_concentration"))
st.plotly_chart(fig_quad, use_container_width=True)
sections.append(("Quadrant Concentration","Scatter of employees by percentile within job level and CTC quadrant."))
images.append((asset_quad,"Quadrant Concentration"))
tables.append(("Quadrant Concentration (sample)", quad_df[["EmployeeID","JobRole","JobLevel","CTC","QuadBucket"]].head(200)))

colA, colB = st.columns(2)
with colA:
    if asset_quad.get("png") and os.path.exists(asset_quad["png"]):
        with open(asset_quad["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG)", f.read(), file_name=os.path.basename(asset_quad["png"]), mime="image/png")
    elif asset_quad.get("html") and os.path.exists(asset_quad["html"]):
        with open(asset_quad["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML)", f.read(), file_name=os.path.basename(asset_quad["html"]), mime="text/html")
    else:
        st.info("No chart asset available for download.")
with colB:
    st.download_button("⬇️ Export Descriptive PDF",
                       create_metric_pdf_bytes("Compensation Quadrant Concentration",
                           "Scatter of employees across quartile-based quadrants. Use to spot concentrations or clusters near min/max/outliers.",
                           quad_df[["EmployeeID","Department","JobRole","JobLevel","CTC","QuadBucket"]].head(200), asset_quad),
                       file_name="quadrant_concentration.pdf", mime="application/pdf")

# Metric D: Bonus % by Job Level
st.subheader("🎁 Average Bonus % of CTC by Job Level")
filtered_df["BonusPct"] = np.where(filtered_df["CTC"]>0, (filtered_df["Bonus"]/filtered_df["CTC"])*100.0, np.nan)
bonus_tbl = filtered_df.groupby("JobLevel", sort=False)["BonusPct"].mean().reset_index().round(2)
st.dataframe(bonus_tbl, use_container_width=True)
fig_bonus = px.bar(bonus_tbl, x="JobLevel", y="BonusPct", title="Avg Bonus % by JobLevel")
asset_bonus = save_plotly_asset(fig_bonus, safe_filename("bonus_pct"))
st.plotly_chart(fig_bonus, use_container_width=True)
sections.append(("Avg Bonus % of CTC","Shows average bonus percent across job levels."))
images.append((asset_bonus,"Avg Bonus % of CTC"))
tables.append(("Avg Bonus % of CTC",bonus_tbl))

colA, colB = st.columns(2)
with colA:
    if asset_bonus.get("png") and os.path.exists(asset_bonus["png"]):
        with open(asset_bonus["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG)", f.read(), file_name=os.path.basename(asset_bonus["png"]), mime="image/png")
    elif asset_bonus.get("html") and os.path.exists(asset_bonus["html"]):
        with open(asset_bonus["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML)", f.read(), file_name=os.path.basename(asset_bonus["html"]), mime="text/html")
    else:
        st.info("No chart asset available for download.")
with colB:
    st.download_button("⬇️ Export Descriptive PDF",
                       create_metric_pdf_bytes("Average Bonus % of CTC by Job Level",
                           "Average bonus percentage across job levels. Useful to check incentive alignment.",
                           bonus_tbl, asset_bonus),
                       file_name="bonus_pct_joblevel.pdf", mime="application/pdf")

# Metric E: Company vs Market Benchmarking
if bench_df is not None:
    st.subheader("📉 Company vs Market (Median CTC)")
    comp_med = filtered_df.groupby("JobLevel", sort=False)["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
    bench_med = bench_df.groupby("JobLevel", sort=False)["MarketMedianCTC"].median().reset_index().rename(columns={"MarketMedianCTC":"MarketMedian"})
    compare = pd.merge(comp_med, bench_med, on="JobLevel", how="outer")
    compare["CompanyMedian"] = pd.to_numeric(compare["CompanyMedian"], errors="coerce").fillna(0)
    compare["MarketMedian"] = pd.to_numeric(compare["MarketMedian"], errors="coerce").fillna(0)
    compare["Gap%"] = np.where(compare["MarketMedian"]>0,
                               (compare["CompanyMedian"] - compare["MarketMedian"]) / compare["MarketMedian"] * 100.0,
                               np.nan).round(2)
    st.dataframe(compare, use_container_width=True)
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(x=compare["JobLevel"], y=compare["CompanyMedian"], name="CompanyMedian"))
    fig_cmp.add_trace(go.Scatter(x=compare["JobLevel"], y=compare["MarketMedian"], name="MarketMedian", mode="lines+markers"))
    fig_cmp.update_layout(title="Company vs Market Median CTC", xaxis_title="JobLevel", yaxis_title="CTC")
    asset_cmp = save_plotly_asset(fig_cmp, safe_filename("company_vs_market"))
    st.plotly_chart(fig_cmp, use_container_width=True)
    sections.append(("Company vs Market","Median comparison by JobLevel."))
    images.append((asset_cmp,"Company vs Market"))
    tables.append(("Company vs Market", compare))

    colA, colB = st.columns(2)
    with colA:
        if asset_cmp.get("png") and os.path.exists(asset_cmp["png"]):
            with open(asset_cmp["png"], "rb") as f:
                st.download_button("⬇️ Export Image (PNG)", f.read(), file_name=os.path.basename(asset_cmp["png"]), mime="image/png")
        elif asset_cmp.get("html") and os.path.exists(asset_cmp["html"]):
            with open(asset_cmp["html"], "rb") as f:
                st.download_button("⬇️ Export Chart (HTML)", f.read(), file_name=os.path.basename(asset_cmp["html"]), mime="text/html")
        else:
            st.info("No chart asset available for download.")
    with colB:
        st.download_button("⬇️ Export Descriptive PDF",
                           create_metric_pdf_bytes("Company vs Market Benchmarking (Median CTC)",
                               "Company median vs market median by job level. Gap% indicates how far from market the company sits.",
                               compare, asset_cmp),
                           file_name="company_vs_market.pdf", mime="application/pdf")

# -----------------------
# Compiled multi-KPI PDF (Cover + TOC + selected sections)
# -----------------------
st.header("📥 Download Reports & Images")
kpi_titles = [s[0] for s in sections]
kpi_check = {}
st.write("Select KPIs to include in compiled report:")
cols = st.columns(3)
for i, title in enumerate(kpi_titles):
    with cols[i % 3]:
        kpi_check[title] = st.checkbox(title, value=False)

if st.button("🧩 Compile Selected KPIs into Single PDF"):
    selected = [k for k, v in kpi_check.items() if v]
    if not selected:
        st.warning("Select at least one KPI to compile.")
    else:
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=18*mm, leftMargin=18*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        story = []
        # Cover
        cover_style = ParagraphStyle("Cover", parent=styles["Title"], fontName=HEADER_FONT, fontSize=22, alignment=1, textColor=TEXT_COLOR)
        story.append(Paragraph("Compensation & Benefits - Compiled Report", cover_style))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}", styles["Normal"]))
        story.append(PageBreak())
        # TOC
        story.append(Paragraph("Table of Contents", styles["Heading2"]))
        for idx, t in enumerate(selected, 1):
            story.append(Paragraph(f'{idx}. {t}', styles["Normal"]))
        story.append(PageBreak())
        # Sections
        for title in selected:
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Spacer(1, 6))
            # include the corresponding table
            table_df = None
            for ttitle, df in tables:
                if ttitle == title:
                    table_df = df
                    break
            if table_df is not None:
                data = [list(table_df.columns)] + table_df.fillna("").values.tolist()
                tstyle = TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.black),
                                    ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke)])
                for i in range(1, len(data)):
                    if i % 2 == 0:
                        tstyle.add("BACKGROUND", (0,i), (-1,i), TABLE_ZEBRA)
                tbl = Table(data, repeatRows=1, hAlign="LEFT")
                tbl.setStyle(tstyle)
                story.append(tbl)
                story.append(Spacer(1, 6))
            # include image if present
            img_info = None
            for ipath, caption in images:
                if caption == title:
                    img_info = ipath
                    break
            if img_info:
                if img_info.get("png") and os.path.exists(img_info["png"]):
                    story.append(RLImage(img_info["png"], width=170*mm, height=90*mm))
                    story.append(Spacer(1,6))
                elif img_info.get("html") and os.path.exists(img_info["html"]):
                    story.append(Paragraph(f"Interactive chart saved as HTML: {os.path.basename(img_info['html'])}", styles["Normal"]))
                    story.append(Spacer(1,6))
                    if img_info.get("error"):
                        story.append(Paragraph(f"Note: PNG export failed with error: {img_info['error']}", styles["Normal"]))
                        story.append(Spacer(1,6))
            # small insight
            if title == "Average CTC by Job Level":
                story.append(Paragraph("Insight: Average CTC highlights central tendency. Check levels with anomalously high averages for outlier headcount or high-salary roles.", styles["Normal"]))
            elif title == "Median CTC by Job Level":
                story.append(Paragraph("Insight: Median reduces outlier effect — useful to compare with Average to detect skew.", styles["Normal"]))
            elif title == "Quartile Placement":
                story.append(Paragraph("Insight: Quartile placement shows concentration of population across pay bands; Outlier counts indicate potential anomalies.", styles["Normal"]))
            elif title == "Quadrant Concentration":
                story.append(Paragraph("Insight: Scatter shows clusters of employees by percentile and absolute pay — useful for targeted calibration.", styles["Normal"]))
            elif title == "Avg Bonus % of CTC":
                story.append(Paragraph("Insight: Bonus % helps check incentive alignment across levels.", styles["Normal"]))
            elif title == "Company vs Market":
                story.append(Paragraph("Insight: Gap% shows competitive vs market placement; negative values indicate being behind market.", styles["Normal"]))
            story.append(PageBreak())

        # Consolidated Conclusions (basic auto-synthesis)
        story.append(Paragraph("Consolidated Conclusions", styles["Heading2"]))
        if any(t == "Company vs Market" for t in kpi_titles) and "Company vs Market" in selected:
            try:
                comp_idx = [t[0] for t in tables].index("Company vs Market")
                comp_df = tables[comp_idx][1]
                comp_tmp = comp_df.dropna(subset=["Gap%"]).sort_values("Gap%")
                if not comp_tmp.empty:
                    worst = comp_tmp.head(3)
                    best = comp_tmp.tail(3)
                    for _, r in worst.iterrows():
                        story.append(Paragraph(f"⚠️ {r['JobLevel']} behind market by {r['Gap%']}%. Consider review.", styles["Normal"]))
                    for _, r in best.iterrows():
                        story.append(Paragraph(f"✅ {r['JobLevel']} ahead of market by {r['Gap%']}%.", styles["Normal"]))
                else:
                    story.append(Paragraph("No benchmark gap data available to summarize.", styles["Normal"]))
            except Exception:
                story.append(Paragraph("No benchmark gap data available to summarize.", styles["Normal"]))
        else:
            story.append(Paragraph("Benchmark data not included in selection — market comparisons not available.", styles["Normal"]))

        doc.build(story,
                  onFirstPage=lambda c,d:(draw_background(c,d), add_page_number(c,d)),
                  onLaterPages=lambda c,d:(draw_background(c,d), add_page_number(c,d)))
        st.download_button("⬇️ Download Compiled PDF", buf.getvalue(), file_name="cb_compiled_report.pdf", mime="application/pdf")

# -----------------------
# Quick image downloads
# -----------------------
st.markdown("### Quick image downloads for slides")
for img_info, caption in images:
    if img_info.get("png") and os.path.exists(img_info["png"]):
        with open(img_info["png"], "rb") as f:
            st.download_button(f"⬇️ {caption}", f.read(), file_name=os.path.basename(img_info["png"]), mime="image/png")
    elif img_info.get("html") and os.path.exists(img_info["html"]):
        with open(img_info["html"], "rb") as f:
            st.download_button(f"⬇️ {caption} (HTML)", f.read(), file_name=os.path.basename(img_info["html"]), mime="text/html")

# -----------------------
# Wrap up
# -----------------------
st.success("Dashboard loaded ✅ Use filters on the left to refine views. Use per-metric PNG/PDF exports or compile multi-KPI PDF.")