# ============================================================
# cb_dashboard.py — Compensation & Benefits Dashboard
# Version: 2.9 (UAT v2 Fixes + New Metrics + Design Refresh)
# Last Updated: 2025-09-29 23:20 IST
# Notes:
# - UAT Observations 1..15 implemented
# - No sidebar — filters on-page per metric
# - Clean headers, numbers in ₹ Lakhs (pure numeric)
# - Kaleido-safe exports with HTML fallback
# - Metrics: Avg, Median, Quartile, Distribution, Quadrant(Scatter/Donut),
#   Bonus%, Company vs Market, Avg by Gender & by Rating
# - Per-metric PNG/HTML/PDF; Compiled PDF with TOC + tabular conclusions
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import os
import math

# ReportLab
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="Compensation & Benefits Dashboard", layout="wide")
TMP_DIR = "temp_charts_cb"
os.makedirs(TMP_DIR, exist_ok=True)

# -----------------------
# Required headers
# -----------------------
EMP_REQUIRED = [
    "EmployeeID", "Gender", "Department", "JobRole",
    "JobLevel", "CTC", "Bonus", "PerformanceRating"
]
BENCH_REQUIRED = ["JobRole", "JobLevel", "MarketMedianCTC"]

# -----------------------
# Visual / PDF constants
# -----------------------
PALETTE = px.colors.qualitative.Prism  # modern, colorful palette
PDF_BG = "#FFFFFF"
HEADER_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"
TABLE_ZEBRA = colors.HexColor("#F7F7F7")
TEXT_COLOR = colors.black

# -----------------------
# Low-level helpers
# -----------------------
def sanitize_anchor(title: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")

def safe_filename(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def readable_lakhs_number(x):
    """Return a float value representing lakhs (no currency symbol)."""
    if pd.isna(x):
        return None
    try:
        val = float(x) / 100000.0
        # round to 2 decimals for tables
        return round(val, 2)
    except Exception:
        return None

def draw_background(canvas, doc):
    "Light frame — keep PDF neutral."
    canvas.saveState()
    canvas.setStrokeColor(colors.black)
    canvas.rect(5, 5, A4[0]-10, A4[1]-10, stroke=1, fill=0)
    canvas.restoreState()

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(280, 15, f"Page {doc.page}")
    canvas.restoreState()

# -----------------------
# Plotly asset saver (kaleido safe)
# -----------------------
def save_plotly_asset(fig, filename_base, width=1200, height=700, scale=2):
    """
    Save plotly figure as PNG (kaleido) where available, else write interactive HTML.
    Returns a dict: {"png": path_or_None, "html": path_or_None}
    """
    base = os.path.join(TMP_DIR, filename_base)
    png_path = base + ".png"
    html_path = base + ".html"
    try:
        # attempt PNG via kaleido
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        with open(png_path, "wb") as f:
            f.write(img_bytes)
        return {"png": png_path, "html": None}
    except Exception:
        # fallback: interactive HTML
        try:
            fig.write_html(html_path)
            return {"png": None, "html": html_path}
        except Exception:
            return {"png": None, "html": None}

def read_binary(path):
    with open(path, "rb") as f:
        return f.read()
# -----------------------
# Templates + How-to Guide (expanded)
# -----------------------
def get_employee_template_csv():
    df = pd.DataFrame(columns=EMP_REQUIRED)
    # add sample rows to guide template visually
    df.loc[0] = ["E1001", "Male", "Finance", "Analyst", "Analyst", 600000, 50000, 3]
    df.loc[1] = ["E1002", "Female", "Engineering", "Senior Manager", "Senior Manager", 4200000, 300000, 2]
    return df.to_csv(index=False)

def get_benchmark_template_csv():
    df = pd.DataFrame(columns=BENCH_REQUIRED)
    df.loc[0] = ["Analyst", "Analyst", 650000]
    df.loc[1] = ["Senior Manager", "Senior Manager", 4400000]
    return df.to_csv(index=False)

def get_howto_markdown_full():
    return """
# How to Upload Data — C&B Dashboard

**Use only the templates provided. Headers are case-sensitive and exact.**

## Internal Compensation Template (Internal_Compensation_Data_Template.csv)
Columns:
- EmployeeID : Unique identifier (string) — Example: E1001
- Gender : Male / Female / Other
- Department : Department name
- JobRole : Job title
- JobLevel : Canonical level (Analyst, Manager, Senior Manager, Director etc.)
- CTC : Annual cash pay in INR (numeric). Example: 600000
- Bonus : Annual bonus in INR (numeric). Example: 50000
- PerformanceRating : Integer 1..5 (1 best, 5 lowest)

## External Benchmarking Template (External_Benchmarking_Data_Template.csv)
Columns:
- JobRole : Role
- JobLevel : Level
- MarketMedianCTC : Median annual pay in INR (numeric)

All outputs in the dashboard & exported tables are standardized to **₹ Lakhs** (pure numeric values) — header indicates units.
"""

def create_howto_pdf_bytes():
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("How to Upload Data — C&B Dashboard",
                           ParagraphStyle("title", parent=styles["Title"], fontName=HEADER_FONT, fontSize=18, alignment=1)))
    story.append(Spacer(1,6))
    body = ParagraphStyle("body", parent=styles["Normal"], fontName=BODY_FONT, fontSize=10, leading=14)
    for para in get_howto_markdown_full().split("\n\n"):
        story.append(Paragraph(para.replace("\n", "<br/>"), body))
        story.append(Spacer(1,6))
    doc.build(story, onFirstPage=lambda c,d:(draw_background(c,d), add_page_number(c,d)),
                   onLaterPages=lambda c,d:(draw_background(c,d), add_page_number(c,d)))
    return buf.getvalue()

# -----------------------
# App header (neutral / not pink)
# -----------------------
st.markdown(f"""
<div style="padding:18px;border-radius:10px;border:1px solid #ddd;text-align:center">
  <h1 style="margin:0;padding:0">📊 Compensation & Benefits Dashboard</h1>
  <p style="margin:6px 0 0 0">Board-ready pay analytics — per-metric filters, exports, and benchmark comparisons.</p>
  <div style="margin-top:8px;display:flex;gap:8px;justify-content:center">
    <img src="https://img.shields.io/badge/version-2.9-blue"/>
    <img src="https://img.shields.io/badge/streamlit-cloud-red?logo=streamlit"/>
    <img src="https://img.shields.io/badge/python-3.10+-yellow?logo=python"/>
  </div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Step 1: Templates & Guide - main page
# -----------------------
st.header("Step 1 — Templates & Guide")
c1, c2 = st.columns(2)
with c1:
    st.download_button("📥 Internal Compensation Data Template",
                       data=get_employee_template_csv(),
                       file_name="Internal_Compensation_Data_Template.csv", mime="text/csv")
with c2:
    st.download_button("📥 External Benchmarking Data Template",
                       data=get_benchmark_template_csv(),
                       file_name="External_Benchmarking_Data_Template.csv", mime="text/csv")

st.download_button("📄 How-to Guide (PDF)",
                   data=create_howto_pdf_bytes(),
                   file_name="How_to_Upload_Guide.pdf",
                   mime="application/pdf")

if not st.checkbox("✅ I downloaded templates + guide"):
    st.info("Please download and review templates before proceeding.")
    st.stop()

# -----------------------
# Step 2: Upload Data
# -----------------------
st.header("Step 2 — Upload Data")
up_col1, up_col2 = st.columns(2)
with up_col1:
    uploaded_file = st.file_uploader("Upload Internal Compensation CSV/XLSX", type=["csv","xlsx"])
with up_col2:
    benchmark_file = st.file_uploader("Upload External Benchmarking CSV/XLSX (optional)", type=["csv","xlsx"])

if not uploaded_file:
    st.warning("Please upload the Internal Compensation file to proceed.")
    st.stop()

def read_input(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")

emp_df = read_input(uploaded_file)
ok, msg = validate_exact_headers(emp_df, EMP_REQUIRED)
if not ok:
    st.error(msg); st.stop()

# Ensure numeric conversion
for c in ["CTC", "Bonus", "PerformanceRating"]:
    emp_df[c] = pd.to_numeric(emp_df[c], errors="coerce")

bench_df = None
if benchmark_file:
    bench_df = read_input(benchmark_file)
    ok_b, msg_b = validate_exact_headers(bench_df, BENCH_REQUIRED)
    if not ok_b:
        st.error(msg_b); st.stop()
    bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce")

# Preview
st.subheader("Preview (first 10 rows)")
st.dataframe(emp_df.head(10), use_container_width=True)
if bench_df is not None:
    st.write("Benchmark preview:")
    st.dataframe(bench_df.head(10), use_container_width=True)
# -----------------------
# Step 3: Filters (on-page helper)
# -----------------------
def metric_filters_ui(df, prefix=""):
    """
    Return filtered df after showing on-page controls.
    prefix: unique prefix for widget keys when multiple filters appear in same page.
    """
    st.markdown("**Filters (applies only to this metric)**")
    cols = st.columns([2,2,2])
    with cols[0]:
        dept = st.selectbox(f"{prefix}Department", ["All"] + sorted(df["Department"].dropna().unique()), key=f"{prefix}_dept")
    with cols[1]:
        roles_all = df[df["Department"]==dept]["JobRole"].dropna().unique().tolist() if dept!="All" else df["JobRole"].dropna().unique().tolist()
        roles = sorted(roles_all)
        sel_roles = st.multiselect(f"{prefix}Job Role", roles, key=f"{prefix}_roles")
    with cols[2]:
        level_vals = sorted(df["JobLevel"].dropna().unique().tolist())
        sel_levels = st.multiselect(f"{prefix}Job Level", level_vals, key=f"{prefix}_levels")
    out = df.copy()
    if dept != "All":
        out = out[out["Department"]==dept]
    if sel_roles:
        out = out[out["JobRole"].isin(sel_roles)]
    if sel_levels:
        out = out[out["JobLevel"].isin(sel_levels)]
    return out

# -----------------------
# Quartile categorizer (used by multiple metrics)
# -----------------------
def make_quartile_categorizer(series):
    q1, q2, q3 = series.quantile([0.25, 0.5, 0.75]).tolist()
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    def categorize(x):
        if pd.isna(x): return "NA"
        if x < low: return "Outlier"
        if x <= q1: return "Q1"
        if x <= q2: return "Q2"
        if x <= q3: return "Q3"
        if x <= high: return "Q4"
        return "Outlier"
    return categorize

# -----------------------
# Per-metric PDF generator
# -----------------------
def create_metric_pdf_bytes(title, description, table_df=None, asset=None):
    """
    asset: dict from save_plotly_asset {"png":..., "html":...}
    table_df: pandas DataFrame with numeric lakhs columns (pure numbers)
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName=BODY_FONT, fontSize=10, leading=13)
    story = []
    # Title + desc
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Spacer(1,6))
    if description:
        story.append(Paragraph(description, body))
        story.append(Spacer(1,6))
    # Table
    if table_df is not None and not table_df.empty:
        data = [list(table_df.columns)] + table_df.fillna("").values.tolist()
        tstyle = TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),
                            ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)])
        for i in range(1,len(data)):
            if i % 2 == 0:
                tstyle.add("BACKGROUND",(0,i),(-1,i),TABLE_ZEBRA)
        tbl = Table(data, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(tstyle)
        story.append(tbl)
        story.append(Spacer(1,8))
    # Chart image
    if asset:
        if asset.get("png") and os.path.exists(asset["png"]):
            story.append(RLImage(asset["png"], width=170*mm, height=90*mm))
        elif asset.get("html") and os.path.exists(asset["html"]):
            story.append(Paragraph(f"Interactive chart saved as HTML attached: {os.path.basename(asset['html'])}", body))
        story.append(Spacer(1,6))
    doc.build(story, onFirstPage=lambda c,d:(draw_background(c,d), add_page_number(c,d)),
                   onLaterPages=lambda c,d:(draw_background(c,d), add_page_number(c,d)))
    return buf.getvalue()

# -----------------------
# Storage for compiled report
# -----------------------
sections = []   # list of (title, description, table_df, asset)
images_for_download = []  # list of dict {title, asset}

# -----------------------
# Helper to standardize table column names and lakhs conversion
# -----------------------
def to_lakhs_table(df, cols_map):
    """
    cols_map: {new_col_name: original_col_name}
    returns DataFrame with new_col_name and numeric lakhs values (float)
    """
    out = pd.DataFrame()
    for new_col, orig_col in cols_map.items():
        if orig_col in df.columns:
            out[new_col] = df[orig_col].apply(readable_lakhs_number)
        else:
            out[new_col] = None
    return out

# -----------------------
# Metric A: Average CTC by JobLevel
# -----------------------
st.subheader("🏷️ Average CTC — Job Level")
dfA = metric_filters_ui(emp_df, prefix="A_")
avg_ctc = dfA.groupby("JobLevel", sort=False)["CTC"].mean().reset_index().rename(columns={"CTC":"AvgCTC"})
avg_ctc["AvgCTC_Lakhs"] = avg_ctc["AvgCTC"].apply(readable_lakhs_number)
avg_table = avg_ctc[["JobLevel","AvgCTC_Lakhs"]].copy()
avg_table.columns = ["JobLevel", "Avg CTC (₹ Lakhs)"]
st.dataframe(avg_table, use_container_width=True)

figA = px.bar(avg_ctc.sort_values("JobLevel"), x="JobLevel", y="AvgCTC",
              color="JobLevel", color_discrete_sequence=PALETTE,
              labels={"AvgCTC":"Avg CTC (INR)"}, title="Average CTC by Job Level")
figA.update_layout(showlegend=False, yaxis_title="CTC (INR)")
assetA = save_plotly_asset(figA, safe_filename("avg_ctc_joblevel"))
st.plotly_chart(figA, use_container_width=True)

# downloads & save section registry
sections.append(("Average CTC by Job Level", "Average CTC across job levels (₹ Lakhs).", avg_table, assetA))
images_for_download.append({"title":"Average CTC by Job Level","asset":assetA})

# Download buttons
colA, colB = st.columns(2)
with colA:
    if assetA.get("png") and os.path.exists(assetA["png"]):
        with open(assetA["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG) — Avg CTC", f.read(), file_name=os.path.basename(assetA["png"]), mime="image/png")
    elif assetA.get("html") and os.path.exists(assetA["html"]):
        with open(assetA["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML) — Avg CTC", f.read(), file_name=os.path.basename(assetA["html"]), mime="text/html")
    else:
        st.info("No chart asset available for Avg CTC.")
with colB:
    st.download_button("⬇️ Export Descriptive PDF — Avg CTC",
                       create_metric_pdf_bytes("Average CTC by Job Level",
                                              "Shows average CTC (in ₹ Lakhs) per Job Level. Use to detect skew vs medians.",
                                              avg_table, assetA),
                       file_name="avg_ctc_joblevel.pdf", mime="application/pdf")
# -----------------------
# Metric B: Median CTC by JobLevel
# -----------------------
st.subheader("📏 Median CTC — Job Level")
dfB = metric_filters_ui(emp_df, prefix="B_")
med_ctc = dfB.groupby("JobLevel", sort=False)["CTC"].median().reset_index().rename(columns={"CTC":"MedianCTC"})
med_ctc["MedianCTC_Lakhs"] = med_ctc["MedianCTC"].apply(readable_lakhs_number)
med_table = med_ctc[["JobLevel","MedianCTC_Lakhs"]].copy()
med_table.columns = ["JobLevel", "Median CTC (₹ Lakhs)"]
st.dataframe(med_table, use_container_width=True)

figB = px.bar(med_ctc, x="JobLevel", y="MedianCTC", color="JobLevel", color_discrete_sequence=PALETTE,
              title="Median CTC by Job Level")
figB.update_layout(showlegend=False, yaxis_title="CTC (INR)")
assetB = save_plotly_asset(figB, safe_filename("median_ctc_joblevel"))
st.plotly_chart(figB, use_container_width=True)

sections.append(("Median CTC by Job Level","Median by Job Level (₹ Lakhs).", med_table, assetB))
images_for_download.append({"title":"Median CTC by Job Level","asset":assetB})

colA, colB = st.columns(2)
with colA:
    if assetB.get("png") and os.path.exists(assetB["png"]):
        with open(assetB["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG) — Median CTC", f.read(), file_name=os.path.basename(assetB["png"]), mime="image/png")
    elif assetB.get("html") and os.path.exists(assetB["html"]):
        with open(assetB["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML) — Median CTC", f.read(), file_name=os.path.basename(assetB["html"]), mime="text/html")
    else:
        st.info("No chart asset available for Median CTC.")
with colB:
    st.download_button("⬇️ Export Descriptive PDF — Median CTC",
                       create_metric_pdf_bytes("Median CTC by Job Level",
                                               "Median CTC per Job Level (₹ Lakhs). Useful to identify central tendency/ skew.",
                                               med_table, assetB),
                       file_name="median_ctc_joblevel.pdf", mime="application/pdf")

# -----------------------
# Metric C: Quartile Placement by Job Level (table) + Distribution (violin)
# -----------------------
st.subheader("📊 Quartile Placement & Distribution")
dfC = metric_filters_ui(emp_df, prefix="C_")
# Quartile table
qcat = make_quartile_categorizer(dfC["CTC"])
quart_rows = []
for lvl, g in dfC.groupby("JobLevel", sort=False):
    cats = g["CTC"].apply(qcat)
    vc = cats.value_counts()
    row = {"JobLevel": lvl, "Count": len(g),
           "Q1": int(vc.get("Q1",0)), "Q2": int(vc.get("Q2",0)),
           "Q3": int(vc.get("Q3",0)), "Q4": int(vc.get("Q4",0)),
           "Outlier": int(vc.get("Outlier",0))}
    quart_rows.append(row)
quart_tbl = pd.DataFrame(quart_rows)
if not quart_tbl.empty:
    totals = quart_tbl.drop(columns=["JobLevel"]).sum()
    total_row = {"JobLevel":"Total", **totals.to_dict()}
    quart_tbl = quart_tbl.append(total_row, ignore_index=True)
st.dataframe(quart_tbl, use_container_width=True)

# Distribution plot (violin + jitter)
figC = go.Figure()
for i, lvl in enumerate(sorted(dfC["JobLevel"].dropna().unique())):
    grp = dfC[dfC["JobLevel"]==lvl]
    if grp.empty: continue
    figC.add_trace(go.Violin(x=[lvl]*len(grp), y=grp["CTC"], name=str(lvl),
                             box_visible=True, meanline_visible=True, marker_color=PALETTE[i % len(PALETTE)]))
figC.update_layout(title="CTC Distribution by JobLevel", yaxis_title="CTC (INR)")
assetC = save_plotly_asset(figC, safe_filename("ctc_distribution"))
st.plotly_chart(figC, use_container_width=True)

sections.append(("Quartile Placement","Quartile counts and distribution.", quart_tbl, assetC))
images_for_download.append({"title":"CTC Distribution by JobLevel","asset":assetC})

colA, colB = st.columns(2)
with colA:
    if assetC.get("png") and os.path.exists(assetC["png"]):
        with open(assetC["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG) — Distribution", f.read(), file_name=os.path.basename(assetC["png"]), mime="image/png")
    elif assetC.get("html") and os.path.exists(assetC["html"]):
        with open(assetC["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML) — Distribution", f.read(), file_name=os.path.basename(assetC["html"]), mime="text/html")
with colB:
    st.download_button("⬇️ Export Descriptive PDF — Distribution",
                       create_metric_pdf_bytes("CTC Distribution by Job Level",
                                               "Violin plots with box & meanline show distribution and skew by job level. Values in ₹ Lakhs in the table.",
                                               quart_tbl, assetC),
                       file_name="ctc_distribution.pdf", mime="application/pdf")

# -----------------------
# Metric D: Quadrant Concentration (Scatter vs Donut)
# -----------------------
st.subheader("⚫ Compensation Quadrant Concentration (Scatter or Donut)")
dfQ = metric_filters_ui(emp_df, prefix="Q_").copy()
q_func = make_quartile_categorizer(dfQ["CTC"])
dfQ["QuartileCat"] = dfQ["CTC"].apply(q_func)
dfQ["PctWithinLevel"] = dfQ.groupby("JobLevel")["CTC"].rank(pct=True) * 100.0

mode = st.radio("Display as", ["Scatter (Percentile vs CTC)", "Donut (Quartile Split)"], index=0, horizontal=True)
if mode.startswith("Scatter"):
    figQ = px.scatter(dfQ, x="PctWithinLevel", y="CTC", color="QuartileCat",
                      color_discrete_sequence=PALETTE,
                      hover_data=["EmployeeID","Department","JobRole","JobLevel"],
                      title="Quadrant Scatter (Percentile within Level vs CTC)")
    figQ.update_layout(xaxis_title="Percentile within JobLevel", yaxis_title="CTC (INR)")
    assetQ = save_plotly_asset(figQ, safe_filename("quadrant_scatter"))
    st.plotly_chart(figQ, use_container_width=True)
else:
    qcounts = dfQ["QuartileCat"].value_counts().reindex(["Q1","Q2","Q3","Q4","Outlier"]).fillna(0)
    donut_df = pd.DataFrame({"Quartile": qcounts.index, "Count": qcounts.values})
    figQ = px.pie(donut_df, names="Quartile", values="Count", hole=0.5, color="Quartile", color_discrete_sequence=PALETTE)
    figQ.update_layout(title="Quartile Share (Donut)")
    assetQ = save_plotly_asset(figQ, safe_filename("quadrant_donut"))
    st.plotly_chart(figQ, use_container_width=True)

sections.append(("Quadrant Concentration", "Scatter of percentiles or Donut of quartile splits.", None, assetQ))
images_for_download.append({"title":"Quadrant Concentration","asset":assetQ})

colA, colB = st.columns(2)
with colA:
    if assetQ.get("png") and os.path.exists(assetQ["png"]):
        with open(assetQ["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG) — Quadrant", f.read(), file_name=os.path.basename(assetQ["png"]), mime="image/png")
    elif assetQ.get("html") and os.path.exists(assetQ["html"]):
        with open(assetQ["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML) — Quadrant", f.read(), file_name=os.path.basename(assetQ["html"]), mime="text/html")
with colB:
    st.download_button("⬇️ Export Descriptive PDF — Quadrant",
                       create_metric_pdf_bytes("Compensation Quadrant Concentration",
                                               "Scatter of employees across quartile-based buckets or donut summarizing quartile counts.",
                                               None, assetQ),
                       file_name="quadrant_concentration.pdf", mime="application/pdf")

# -----------------------
# Metric E: Bonus % by JobLevel
# -----------------------
st.subheader("🎁 Average Bonus % of CTC by Job Level")
dfD = metric_filters_ui(emp_df, prefix="D_")
dfD = dfD.copy()
dfD["BonusPct"] = np.where(dfD["CTC"]>0, (dfD["Bonus"]/dfD["CTC"])*100.0, np.nan)
bonus_tbl = dfD.groupby("JobLevel", sort=False)["BonusPct"].mean().reset_index()
bonus_tbl["BonusPct"] = bonus_tbl["BonusPct"].round(2)
bonus_tbl.columns = ["JobLevel", "Avg Bonus %"]
st.dataframe(bonus_tbl, use_container_width=True)

figD = px.bar(bonus_tbl, x="JobLevel", y="Avg Bonus %", color="JobLevel", color_discrete_sequence=PALETTE, title="Avg Bonus % by Job Level")
assetD = save_plotly_asset(figD, safe_filename("bonus_pct"))
st.plotly_chart(figD, use_container_width=True)
sections.append(("Avg Bonus % of CTC","Average bonus percent across job levels.", bonus_tbl, assetD))
images_for_download.append({"title":"Avg Bonus % of CTC","asset":assetD})

colA, colB = st.columns(2)
with colA:
    if assetD.get("png") and os.path.exists(assetD["png"]):
        with open(assetD["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG) — Bonus %", f.read(), file_name=os.path.basename(assetD["png"]), mime="image/png")
    elif assetD.get("html") and os.path.exists(assetD["html"]):
        with open(assetD["html"], "rb") as f:
            st.download_button("⬇️ Export Chart (HTML) — Bonus %", f.read(), file_name=os.path.basename(assetD["html"]), mime="text/html")
with colB:
    st.download_button("⬇️ Export Descriptive PDF — Bonus %",
                       create_metric_pdf_bytes("Average Bonus % of CTC by Job Level",
                                               "Average bonus percent across job levels. Useful to check incentive alignment.",
                                               bonus_tbl, assetD),
                       file_name="bonus_pct_joblevel.pdf", mime="application/pdf")
# -----------------------
# Metric F: Company vs Market Benchmarking
# -----------------------
if bench_df is not None:
    st.subheader("📉 Company vs Market (Median CTC)")
    dfF = metric_filters_ui(emp_df, prefix="F_")
    comp_med = dfF.groupby("JobLevel", sort=False)["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
    bench_med = bench_df.groupby("JobLevel", sort=False)["MarketMedianCTC"].median().reset_index().rename(columns={"MarketMedianCTC":"MarketMedian"})
    compare = pd.merge(comp_med, bench_med, on="JobLevel", how="outer")
    compare["CompanyMedian_Lakhs"] = compare["CompanyMedian"].apply(readable_lakhs_number)
    compare["MarketMedian_Lakhs"] = compare["MarketMedian"].apply(readable_lakhs_number)
    compare["Gap %"] = np.where(compare["MarketMedian"]>0,
                               (compare["CompanyMedian"] - compare["MarketMedian"]) / compare["MarketMedian"] * 100.0,
                               np.nan).round(2)
    cmp_tbl = compare[["JobLevel","CompanyMedian_Lakhs","MarketMedian_Lakhs","Gap %"]]
    cmp_tbl.columns = ["JobLevel","Company Median (₹ Lakhs)","Market Median (₹ Lakhs)","Gap %"]
    st.dataframe(cmp_tbl, use_container_width=True)

    figF = go.Figure()
    figF.add_trace(go.Bar(x=compare["JobLevel"], y=compare["CompanyMedian"], name="CompanyMedian"))
    figF.add_trace(go.Scatter(x=compare["JobLevel"], y=compare["MarketMedian"], name="MarketMedian", mode="lines+markers"))
    figF.update_layout(title="Company vs Market Median CTC", yaxis_title="CTC (INR)")
    assetF = save_plotly_asset(figF, safe_filename("company_vs_market"))
    st.plotly_chart(figF, use_container_width=True)

    sections.append(("Company vs Market", "Company vs Market median pay comparison (₹ Lakhs).", cmp_tbl, assetF))
    images_for_download.append({"title":"Company vs Market","asset":assetF})

    colA, colB = st.columns(2)
    with colA:
        if assetF.get("png") and os.path.exists(assetF["png"]):
            with open(assetF["png"], "rb") as f:
                st.download_button("⬇️ Export Image (PNG) — Company vs Market", f.read(), file_name=os.path.basename(assetF["png"]), mime="image/png")
    with colB:
        st.download_button("⬇️ Export Descriptive PDF — Company vs Market",
                           create_metric_pdf_bytes("Company vs Market Benchmarking (Median CTC)",
                                                   "Gap % shows where company lags or leads market by Job Level.",
                                                   cmp_tbl, assetF),
                           file_name="company_vs_market.pdf", mime="application/pdf")

# -----------------------
# Metric G: Avg CTC by Gender & JobLevel
# -----------------------
st.subheader("👫 Avg CTC — Gender x Job Level")
dfG = metric_filters_ui(emp_df, prefix="G_")
g_tbl = dfG.groupby(["JobLevel","Gender"], sort=False)["CTC"].mean().reset_index()
g_tbl["AvgCTC_Lakhs"] = g_tbl["CTC"].apply(readable_lakhs_number)
pivot_g = g_tbl.pivot(index="JobLevel", columns="Gender", values="AvgCTC_Lakhs").fillna(0).round(2).reset_index()
st.dataframe(pivot_g, use_container_width=True)

figG = px.bar(g_tbl, x="JobLevel", y="CTC", color="Gender",
              barmode="group", color_discrete_sequence=PALETTE,
              title="Avg CTC by Gender & Job Level")
figG.update_layout(yaxis_title="CTC (INR)")
assetG = save_plotly_asset(figG, safe_filename("avg_ctc_gender"))
st.plotly_chart(figG, use_container_width=True)

sections.append(("Avg CTC by Gender & Job Level","Avg CTC by gender split across job levels.", pivot_g, assetG))
images_for_download.append({"title":"Avg CTC by Gender & Job Level","asset":assetG})

colA, colB = st.columns(2)
with colA:
    if assetG.get("png") and os.path.exists(assetG["png"]):
        with open(assetG["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG) — Gender", f.read(), file_name=os.path.basename(assetG["png"]), mime="image/png")
with colB:
    st.download_button("⬇️ Export Descriptive PDF — Gender",
                       create_metric_pdf_bytes("Avg CTC by Gender & Job Level",
                                               "Highlights pay gap trends between genders at each job level.",
                                               pivot_g, assetG),
                       file_name="avg_ctc_gender.pdf", mime="application/pdf")

# -----------------------
# Metric H: Avg CTC by Performance Rating & JobLevel
# -----------------------
st.subheader("⭐ Avg CTC — Rating x Job Level")
dfH = metric_filters_ui(emp_df, prefix="H_")
r_tbl = dfH.groupby(["JobLevel","PerformanceRating"], sort=False)["CTC"].mean().reset_index()
r_tbl["AvgCTC_Lakhs"] = r_tbl["CTC"].apply(readable_lakhs_number)
pivot_r = r_tbl.pivot(index="JobLevel", columns="PerformanceRating", values="AvgCTC_Lakhs").fillna(0).round(2).reset_index()
pivot_r.columns = ["JobLevel"] + [f"Rater {c}" for c in pivot_r.columns[1:]]
st.dataframe(pivot_r, use_container_width=True)

figH = px.bar(r_tbl, x="JobLevel", y="CTC", color="PerformanceRating",
              barmode="group", color_discrete_sequence=PALETTE,
              title="Avg CTC by Performance Rating & Job Level")
figH.update_layout(yaxis_title="CTC (INR)")
assetH = save_plotly_asset(figH, safe_filename("avg_ctc_rating"))
st.plotly_chart(figH, use_container_width=True)

sections.append(("Avg CTC by Rating & Job Level","Avg CTC by rating split across job levels.", pivot_r, assetH))
images_for_download.append({"title":"Avg CTC by Rating & Job Level","asset":assetH})

colA, colB = st.columns(2)
with colA:
    if assetH.get("png") and os.path.exists(assetH["png"]):
        with open(assetH["png"], "rb") as f:
            st.download_button("⬇️ Export Image (PNG) — Rating", f.read(), file_name=os.path.basename(assetH["png"]), mime="image/png")
with colB:
    st.download_button("⬇️ Export Descriptive PDF — Rating",
                       create_metric_pdf_bytes("Avg CTC by Rating & Job Level",
                                               "Shows how average pay varies across performance ratings per job level (1=highest, 5=lowest).",
                                               pivot_r, assetH),
                       file_name="avg_ctc_rating.pdf", mime="application/pdf")
# -----------------------
# Compiled multi-KPI PDF
# -----------------------
st.header("📥 Download Reports")
st.write("Select metrics to include in compiled PDF report:")

kpi_check = {}
cols = st.columns(3)
for i,(title,_,_,_) in enumerate(sections):
    with cols[i % 3]:
        kpi_check[title] = st.checkbox(title, value=False)

if st.button("🧾 Compile Selected Report"):
    sel = [s for s in sections if kpi_check.get(s[0])]
    if not sel:
        st.warning("Select at least one metric.")
    else:
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=18*mm, leftMargin=18*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        story = []

        # Cover
        story.append(Paragraph("Compensation & Benefits — Compiled Report", styles["Title"]))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}", styles["Normal"]))
        story.append(PageBreak())

        # TOC
        story.append(Paragraph("Table of Contents", styles["Heading2"]))
        for i,(title,_,_,_) in enumerate(sel,1):
            anchor = sanitize_anchor(title)
            story.append(Paragraph(f"{i}. {title}", styles["Normal"]))
        story.append(PageBreak())

        # Sections
        for title, desc, tbl, asset in sel:
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Spacer(1,6))
            story.append(Paragraph(desc, styles["Normal"]))
            story.append(Spacer(1,6))
            if tbl is not None and not tbl.empty:
                data = [list(tbl.columns)] + tbl.fillna("").values.tolist()
                tstyle = TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),
                                    ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)])
                for i in range(1,len(data)):
                    if i % 2 == 0:
                        tstyle.add("BACKGROUND",(0,i),(-1,i),TABLE_ZEBRA)
                story.append(Table(data, repeatRows=1, hAlign="LEFT"))
                story.append(Spacer(1,6))
            if asset and asset.get("png") and os.path.exists(asset["png"]):
                story.append(RLImage(asset["png"], width=170*mm, height=90*mm))
            story.append(PageBreak())

        # Consolidated conclusions table
        story.append(Paragraph("Consolidated Conclusions", styles["Heading2"]))
        conc_data = [["Metric","Actionable Insight"]]
        for title, desc, tbl, asset in sel:
            if title=="Company vs Market" and tbl is not None and "Gap %" in tbl.columns:
                for _,r in tbl.iterrows():
                    if not pd.isna(r["Gap %"]):
                        if r["Gap %"] < 0:
                            conc_data.append([r["JobLevel"], f"⚠️ Below Market by {r['Gap %']}%"])
                        elif r["Gap %"] > 0:
                            conc_data.append([r["JobLevel"], f"✅ Above Market by {r['Gap %']}%"])
            else:
                conc_data.append([title,"Review visual/table for insights."])
        story.append(Table(conc_data, repeatRows=1, hAlign="LEFT"))
        doc.build(story, onFirstPage=lambda c,d:(draw_background(c,d), add_page_number(c,d)),
                       onLaterPages=lambda c,d:(draw_background(c,d), add_page_number(c,d)))
        st.download_button("⬇️ Download Compiled PDF", buf.getvalue(),
                           file_name="cb_dashboard_report.pdf", mime="application/pdf")

# -----------------------
# Quick image downloads
# -----------------------
st.subheader("📸 Quick Image Downloads")
for item in images_for_download:
    asset = item["asset"]
    if asset.get("png") and os.path.exists(asset["png"]):
        with open(asset["png"], "rb") as f:
            st.download_button(f"⬇️ {item['title']}", f.read(), file_name=os.path.basename(asset["png"]), mime="image/png")

# -----------------------
# Wrap
# -----------------------
st.success("Dashboard loaded ✅ All metrics available with per-metric filters, exports, and compiled PDF option.")