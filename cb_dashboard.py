# ============================================================
# cb_dashboard.py — Compensation & Benefits Dashboard
# Version: 2.1 (Post Step A + Step B + Background + Outliers + Quadrant Fix)
# Last Updated: 2025-09-29 10:00 IST
# Notes:
# - Implements Step A (PDF styling, pastel rose bg)
# - Implements Step B (12 asks, strict headers)
# - Dependent filters (Department → JobRole)
# - Per-metric PNG + PDF export
# - Multi-metric compiled PDF (cover + TOC + insights)
# - Quartile placement, Company vs Market gap analysis
# - Outliers grouped, Quadrant categories fixed
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# import seaborn as sns   # unused, kept for reference
from io import BytesIO
from datetime import datetime
import os
# import base64            # unused
# import tempfile          # unused
# import json              # unused
# import math              # unused

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
                   layout="wide", initial_sidebar_state="collapsed")
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
# STEP A: PDF styling constants
# -----------------------
PDF_BG_COLOR = "#F5E2E3"  # pastel rose
PDF_BORDER_COLOR = colors.black
HEADER_FONT = "Helvetica-Bold"
SUBHEADER_FONT = "Helvetica"
BODY_FONT = "Helvetica"
HIGHLIGHT_BG = colors.yellow
TABLE_ZEBRA = colors.HexColor("#ECEBE8")
TABLE_HEADER_BG = colors.HexColor("#FFFFFF")
TEXT_COLOR = colors.black

def draw_background(canvas, doc):
    """Draw pastel background + black border on every page."""
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(PDF_BG_COLOR))
    canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    # Border
    canvas.setStrokeColor(PDF_BORDER_COLOR)
    canvas.rect(5, 5, A4[0]-10, A4[1]-10, stroke=1, fill=0)
    canvas.restoreState()

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(280, 15, f"Page {doc.page}")
    canvas.restoreState()

# -----------------------
# Utility helpers
# -----------------------
def save_plotly_png(fig, filename, width=1200, height=700, scale=2):
    """Save plotly figure as PNG and return path."""
    p = os.path.join(TMP_DIR, filename)
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    with open(p, "wb") as f:
        f.write(img_bytes)
    return p

def validate_exact_headers(df, required_cols):
    cols = list(df.columns)
    if cols == required_cols:
        return True, "OK"
    else:
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
    md = f"""
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
    return md

def create_howto_pdf_bytes():
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=18*mm, leftMargin=18*mm,
        topMargin=20*mm, bottomMargin=20*mm
    )
    styles = getSampleStyleSheet()
    story = []
    title_style = ParagraphStyle(
        "title", parent=styles["Title"],
        fontName=HEADER_FONT, fontSize=20,
        alignment=1, textColor=TEXT_COLOR
    )
    story.append(Paragraph("How to Upload Data — C&B Dashboard", title_style))
    story.append(Spacer(1, 8))
    body_style = ParagraphStyle(
        "body", parent=styles["Normal"],
        fontName=BODY_FONT, fontSize=10,
        leading=14, textColor=TEXT_COLOR
    )
    md = get_howto_markdown()
    for para in md.split("\n\n"):
        story.append(Paragraph(para.replace("\n","<br/>"), body_style))
        story.append(Spacer(1,6))
    doc.build(story, onFirstPage=draw_background, onLaterPages=draw_background)
    pdf_bytes = buf.getvalue(); buf.close()
    return pdf_bytes

# -----------------------
# Sidebar: Templates & Guide
# -----------------------
st.sidebar.header("Step 1: Templates & Guide")
colA, colB = st.sidebar.columns(2)
with colA:
    st.sidebar.download_button("📥 Employee Template",
        data=get_employee_template_csv(),
        file_name="Employee_Template.csv", mime="text/csv")
with colB:
    st.sidebar.download_button("📥 Benchmark Template",
        data=get_benchmark_template_csv(),
        file_name="Benchmark_Template.csv", mime="text/csv")

st.sidebar.download_button("📄 How-to Guide (PDF)",
    data=create_howto_pdf_bytes(),
    file_name="How_to_Upload_Guide.pdf", mime="application/pdf")

confirm_download = st.sidebar.checkbox(
    "✅ I downloaded templates + guide", value=False
)
if not confirm_download:
    st.sidebar.info("Please download templates & guide before proceeding.")
    st.stop()

# -----------------------
# Upload & Validation
# -----------------------
st.header("Upload Data (strict templates required)")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📂 Upload Employee Compensation", type=["csv","xlsx"])
with col2:
    benchmark_file = st.file_uploader("📂 Upload Benchmarking [optional]", type=["csv","xlsx"])

if not uploaded_file:
    st.info("Please upload the Employee Compensation file.")
    st.stop()

def read_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")

emp_df = read_file(uploaded_file)
ok, msg = validate_exact_headers(emp_df, EMP_REQUIRED)
if not ok:
    st.error(msg); st.stop()

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

# Preview
st.subheader("👀 Preview Data")
st.dataframe(emp_df.head(10), use_container_width=True)
if bench_df is not None:
    st.write("Benchmark Preview:")
    st.dataframe(bench_df.head(10), use_container_width=True)

# -----------------------
# Filters (Dept → JobRole)
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
# Quartile categorizer (with outliers grouped)
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
# Helpers for per-metric PDF
# -----------------------
def create_metric_pdf_section(story, title, description, table_df=None, chart_path=None, styles=None):
    story.append(Paragraph(title, styles["Heading2"]))
    if description:
        story.append(Paragraph(description, styles["BodyText"]))
        story.append(Spacer(1,6))
    if table_df is not None:
        data = [list(table_df.columns)] + table_df.head(200).fillna("").values.tolist()
        tstyle = TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),
                             ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)])
        for i in range(1,len(data)):
            if i % 2 == 0:
                tstyle.add("BACKGROUND",(0,i),(-1,i),TABLE_ZEBRA)
        tbl = Table(data, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(tstyle)
        story.append(tbl)
        story.append(Spacer(1,6))
    if chart_path and os.path.exists(chart_path):
        story.append(RLImage(chart_path, width=170*mm, height=90*mm))
        story.append(Spacer(1,6))
    story.append(PageBreak())

# -----------------------
# METRICS IMPLEMENTATION
# -----------------------
sections, images, tables = [], [], []

# Metric A: Average CTC by Job Level
st.subheader("🏷️ Average CTC by Job Level")
avg_ctc_jl = filtered_df.groupby("JobLevel")["CTC"].mean().reset_index()
avg_ctc_jl["CTC_fmt"] = avg_ctc_jl["CTC"].apply(readable_currency)
st.dataframe(avg_ctc_jl[["JobLevel","CTC_fmt"]], use_container_width=True)
fig_avg = px.bar(avg_ctc_jl, x="JobLevel", y="CTC", title="Average CTC by Job Level")
png_avg = save_plotly_png(fig_avg, safe_filename("avg_ctc_joblevel"))
st.plotly_chart(fig_avg, use_container_width=True)
col1, col2 = st.columns(2)
with col1:
    st.download_button("📸 Export PNG", open(png_avg,"rb").read(), file_name=os.path.basename(png_avg), mime="image/png")
with col2:
    if st.button("📄 PDF: Avg CTC by Job Level"):
        buf=BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4); styles=getSampleStyleSheet(); story=[]
        create_metric_pdf_section(story,"Average CTC by Job Level","Average CTC across job levels.",avg_ctc_jl,png_avg,styles)
        doc.build(story,onFirstPage=draw_background,onLaterPages=draw_background)
        st.download_button("⬇️ Download PDF",buf.getvalue(),"avg_ctc_joblevel.pdf","application/pdf")
sections.append(("Average CTC by Job Level","Shows average pay across job levels."))
images.append((png_avg,"Average CTC by Job Level"))
tables.append(("Average CTC by Job Level",avg_ctc_jl))

# Metric B: Median CTC by Job Level
st.subheader("📏 Median CTC by Job Level")
med_ctc_jl = filtered_df.groupby("JobLevel")["CTC"].median().reset_index().rename(columns={"CTC":"MedianCTC"})
med_ctc_jl["MedianCTC_fmt"] = med_ctc_jl["MedianCTC"].apply(readable_currency)
st.dataframe(med_ctc_jl[["JobLevel","MedianCTC_fmt"]], use_container_width=True)
fig_med = px.bar(med_ctc_jl, x="JobLevel", y="MedianCTC", title="Median CTC by Job Level")
png_med = save_plotly_png(fig_med, safe_filename("median_ctc_joblevel"))
st.plotly_chart(fig_med, use_container_width=True)
col1, col2 = st.columns(2)
with col1:
    st.download_button("📸 Export PNG", open(png_med,"rb").read(), file_name=os.path.basename(png_med), mime="image/png")
with col2:
    if st.button("📄 PDF: Median CTC by Job Level"):
        buf=BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4); styles=getSampleStyleSheet(); story=[]
        create_metric_pdf_section(story,"Median CTC by Job Level","Median CTC across job levels.",med_ctc_jl,png_med,styles)
        doc.build(story,onFirstPage=draw_background,onLaterPages=draw_background)
        st.download_button("⬇️ Download PDF",buf.getvalue(),"median_ctc_joblevel.pdf","application/pdf")
sections.append(("Median CTC by Job Level","Shows median pay across job levels."))
images.append((png_med,"Median CTC by Job Level"))
tables.append(("Median CTC by Job Level",med_ctc_jl))

# Metric C: Quartile Placement by Job Level
st.subheader("📊 Quartile Placement by Job Level")
def build_quartile_table(df, group_col="JobLevel"):
    rows=[]
    for lvl,g in df.groupby(group_col):
        cat_func,_=make_quartile_categorizer(g["CTC"])
        cats=g["CTC"].apply(cat_func)
        vc=cats.value_counts()
        row={"JobLevel":lvl,"Count":len(g),
             "Q1":vc.get("Q1",0),"Q2":vc.get("Q2",0),
             "Q3":vc.get("Q3",0),"Q4":vc.get("Q4",0),
             "Outlier":vc.get("Outlier",0)}
        rows.append(row)
    out=pd.DataFrame(rows)
    if not out.empty:
        totals=out.drop(columns=["JobLevel"]).sum()
        out.loc[len(out)]={"JobLevel":"Total",**totals}
        grand=totals["Count"] if "Count" in totals else 0
        perc={c:round(100*totals[c]/grand,2) if grand>0 else 0 for c in ["Q1","Q2","Q3","Q4","Outlier"]}
        perc.update({"Count":grand}); out.loc[len(out)]={"JobLevel":"Total (%)",**perc}
    return out
quartile_tbl=build_quartile_table(filtered_df)
st.dataframe(quartile_tbl,use_container_width=True)
fig_box=px.box(filtered_df,x="JobLevel",y="CTC",points="all",title="CTC Distribution by JobLevel")
png_box=save_plotly_png(fig_box,safe_filename("quartile_box"))
st.plotly_chart(fig_box,use_container_width=True)
st.download_button("📸 Export PNG - Quartile",open(png_box,"rb").read(),file_name=os.path.basename(png_box),mime="image/png")
sections.append(("Quartile Placement","Distribution by quartiles + outliers per Job Level."))
images.append((png_box,"Quartile Placement"))
tables.append(("Quartile Placement",quartile_tbl))

# Metric C2: Scatter Quadrant (per JobLevel)
st.subheader("⚫ Compensation Quadrant Concentration")
def add_quadrant(df):
    df=df.copy()
    cat_func,_=make_quartile_categorizer(df["CTC"])
    df["QuartileCat"]=df["CTC"].apply(cat_func)
    return df
quad_df=filtered_df.groupby("JobLevel",group_keys=False).apply(add_quadrant)
def quad_bucket(cat):
    if cat=="Q1": return "Q1"
    if cat=="Q2": return "Q2"
    if cat=="Q3": return "Q3"
    if cat=="Q4": return "Q4"
    return "Outlier"
quad_df["QuadBucket"]=quad_df["QuartileCat"].apply(quad_bucket)
fig_quad=px.scatter(quad_df,x="JobLevel",y="CTC",color="QuadBucket",
                    hover_data=["EmployeeID","Department","JobRole"],
                    title="Quadrant Concentration per JobLevel")
png_quad=save_plotly_png(fig_quad,safe_filename("quadrant"))
st.plotly_chart(fig_quad,use_container_width=True)
st.download_button("📸 Export PNG - Quadrant",open(png_quad,"rb").read(),
                   file_name=os.path.basename(png_quad),mime="image/png")
sections.append(("Quadrant Concentration","Scatter of employees in quartiles per JobLevel."))
images.append((png_quad,"Quadrant Concentration"))
tables.append(("Quadrant Concentration",quad_df[["EmployeeID","JobLevel","CTC","QuadBucket"]].head(100)))

# Metric D: Bonus % by Job Level
st.subheader("🎁 Average Bonus % of CTC by Job Level")
filtered_df["BonusPct"]=np.where(filtered_df["CTC"]>0,(filtered_df["Bonus"]/filtered_df["CTC"])*100,np.nan)
bonus_tbl=filtered_df.groupby("JobLevel")["BonusPct"].mean().reset_index().round(2)
st.dataframe(bonus_tbl,use_container_width=True)
fig_bonus=px.bar(bonus_tbl,x="JobLevel",y="BonusPct",title="Avg Bonus % by JobLevel")
png_bonus=save_plotly_png(fig_bonus,safe_filename("bonus_pct"))
st.plotly_chart(fig_bonus,use_container_width=True)
st.download_button("📸 Export PNG - Bonus",open(png_bonus,"rb").read(),
                   file_name=os.path.basename(png_bonus),mime="image/png")
sections.append(("Avg Bonus % of CTC","Shows avg bonus % by JobLevel."))
images.append((png_bonus,"Avg Bonus % of CTC"))
tables.append(("Avg Bonus % of CTC",bonus_tbl))

# Metric E: Company vs Market Benchmarking
if bench_df is not None:
    st.subheader("📉 Company vs Market (Median CTC)")
    comp_med=filtered_df.groupby("JobLevel")["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
    bench_med=bench_df.groupby("JobLevel")["MarketMedianCTC"].median().reset_index().rename(columns={"MarketMedianCTC":"MarketMedian"})
    compare=pd.merge(comp_med,bench_med,on="JobLevel",how="outer").fillna(0)
    compare["Gap%"]=np.where(compare["MarketMedian"]>0,(compare["CompanyMedian"]-compare["MarketMedian"])/compare["MarketMedian"]*100,np.nan).round(2)
    st.dataframe(compare,use_container_width=True)
    fig_cmp=go.Figure()
    fig_cmp.add_trace(go.Bar(x=compare["JobLevel"],y=compare["CompanyMedian"],name="CompanyMedian"))
    fig_cmp.add_trace(go.Scatter(x=compare["JobLevel"],y=compare["MarketMedian"],name="MarketMedian",mode="lines+markers"))
    fig_cmp.update_layout(title="Company vs Market Median CTC",xaxis_title="JobLevel",yaxis_title="CTC")
    png_cmp=save_plotly_png(fig_cmp,safe_filename("company_vs_market"))
    st.plotly_chart(fig_cmp,use_container_width=True)
    st.download_button("📸 Export PNG - Benchmark",open(png_cmp,"rb").read(),
                       file_name=os.path.basename(png_cmp),mime="image/png")
    sections.append(("Company vs Market","Median comparison by JobLevel."))
    images.append((png_cmp,"Company vs Market"))
    tables.append(("Company vs Market",compare))
# -----------------------
# Downloads area (StepB.12)
# -----------------------
st.header("📥 Download Reports & Images")

# Build KPI grid from sections
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
        # Cover
        cover_style=ParagraphStyle("Cover",parent=styles["Title"],fontName=HEADER_FONT,fontSize=22,
                                   alignment=1,textColor=TEXT_COLOR)
        story.append(Paragraph("Compensation & Benefits - Compiled Report",cover_style))
        story.append(Spacer(1,6))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}",styles["Normal"]))
        story.append(PageBreak())
        # TOC
        story.append(Paragraph("Table of Contents",styles["Heading2"]))
        for idx,t in enumerate(selected,1):
            story.append(Paragraph(f"{idx}. {t}",styles["Normal"]))
        story.append(PageBreak())
        # Sections
        for idx,title in enumerate(selected,1):
            story.append(Paragraph(title,styles["Heading2"]))
            table_df=None; image_path=None
            for ttitle,df in tables:
                if ttitle==title: table_df=df; break
            for ipath,caption in images:
                if caption==title: image_path=ipath; break
            create_metric_pdf_section(story,title,f"Auto-generated section for {title}.",table_df,image_path,styles)
        # Consolidated Conclusions
        story.append(Paragraph("Consolidated Conclusions",styles["Heading2"]))
        if any("Company vs Market" in t for t in kpi_titles):
            tmp=tables[[t[0] for t in tables].index("Company vs Market")][1]
            tmp=tmp.dropna(subset=["Gap%"]).sort_values("Gap%")
            worst=tmp.head(3); best=tmp.tail(3)
            for _,r in worst.iterrows():
                story.append(Paragraph(f"⚠️ {r['JobLevel']} behind market by {r['Gap%']}%.",styles["Normal"]))
            for _,r in best.iterrows():
                story.append(Paragraph(f"✅ {r['JobLevel']} ahead of market by {r['Gap%']}%.",styles["Normal"]))
        else:
            story.append(Paragraph("No benchmark data to derive conclusions.",styles["Normal"]))
        doc.build(story,onFirstPage=draw_background,onLaterPages=draw_background)
        st.download_button("⬇️ Download Compiled PDF",buf.getvalue(),
                           file_name="cb_compiled_report.pdf",mime="application/pdf")

# Quick image downloads
st.markdown("### Quick image downloads for slides")
imgs_map={caption:path for path,caption in images}
for caption,path in imgs_map.items():
    if os.path.exists(path):
        with open(path,"rb") as f:
            st.download_button(f"⬇️ {caption}",f.read(),
                               file_name=os.path.basename(path),mime="image/png")

# End of app
st.success("Dashboard loaded ✅ Use filters on the left to drive all metrics. \
Export PNGs / per-metric PDFs, or compile selected KPIs into a single styled PDF.")