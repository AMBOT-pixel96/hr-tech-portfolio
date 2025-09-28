# ============================================================
# cb_dashboard.py — Compensation & Benefits Dashboard
# Version: 3.0 (Patched: validate_exact_headers + concat fix)
# Last Updated: 2025-09-30 00:10 IST
# Notes:
# - UAT Observations 1..15 implemented
# - No sidebar — filters on-page per metric
# - Clean headers, numbers in ₹ Lakhs (pure numeric)
# - Kaleido-safe exports with HTML fallback
# - Metrics: Avg, Median, Quartile, Distribution, Quadrant, Bonus%,
#   Company vs Market, Gender x Level, Rating x Level
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

# ReportLab
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
PALETTE = px.colors.qualitative.Prism
PDF_BG = "#FFFFFF"
HEADER_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"
TABLE_ZEBRA = colors.HexColor("#F7F7F7")
TEXT_COLOR = colors.black

# -----------------------
# Helpers
# -----------------------
def validate_exact_headers(df_or_cols, required_cols):
    if hasattr(df_or_cols, "columns"):
        cols = list(df_or_cols.columns)
    else:
        cols = list(df_or_cols)
    if cols == required_cols:
        return True, "OK"
    return False, f"Header mismatch. Expected {required_cols}, found {cols}"

def sanitize_anchor(title: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")

def safe_filename(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def readable_lakhs_number(x):
    if pd.isna(x):
        return None
    try:
        return round(float(x) / 100000.0, 2)
    except Exception:
        return None

def draw_background(canvas, doc):
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
# Plotly asset saver
# -----------------------
def save_plotly_asset(fig, filename_base, width=1200, height=700, scale=2):
    base = os.path.join(TMP_DIR, filename_base)
    png_path = base + ".png"
    html_path = base + ".html"
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        with open(png_path, "wb") as f:
            f.write(img_bytes)
        return {"png": png_path, "html": None}
    except Exception:
        try:
            fig.write_html(html_path)
            return {"png": None, "html": html_path}
        except Exception:
            return {"png": None, "html": None}

def read_binary(path):
    with open(path, "rb") as f:
        return f.read()

# -----------------------
# Templates + How-to Guide
# -----------------------
def get_employee_template_csv():
    df = pd.DataFrame(columns=EMP_REQUIRED)
    df.loc[0] = ["E1001", "Male", "Finance", "Analyst", "Analyst", 600000, 50000, 3]
    return df.to_csv(index=False)

def get_benchmark_template_csv():
    df = pd.DataFrame(columns=BENCH_REQUIRED)
    df.loc[0] = ["Analyst", "Analyst", 650000]
    return df.to_csv(index=False)

def get_howto_markdown_full():
    return """
# 📘 How to Upload Data — User Guide

This dashboard requires data to be uploaded in **strictly defined templates**.  
Please review these rules carefully before preparing files.

---

## ✅ 1. General Instructions
- 📥 Download the official templates from the dashboard (Employee & Benchmark).  
- ✏️ Fill in your organization’s data directly in those templates.  
- ❌ Do not rename headers, add/remove/reorder columns, or merge cells.  
- 💾 Files must be saved in **.xlsx** format (Excel).  
- ⚠️ Any mismatch in headers will block the upload.

---

## 🧑‍💼 2. Internal Compensation Template

**Required Columns (must match exactly):**
1. EmployeeID  
2. Gender  
3. Department  
4. JobRole  
5. JobLevel  
6. CTC  
7. Bonus  
8. PerformanceRating  

**Column Descriptions with Examples:**

| Column            | Description                                         | Example   |
|-------------------|-----------------------------------------------------|-----------|
| EmployeeID        | Unique employee identifier (string)                 | E1001     |
| Gender            | Male / Female / Other                               | Male      |
| Department        | Functional unit / business vertical                 | Finance   |
| JobRole           | Standardized job role                               | Analyst   |
| JobLevel          | Job grade / level                                   | Analyst   |
| CTC               | Annual Cost to Company (₹, numeric)                 | 600000    |
| Bonus             | Annual bonus / variable pay (₹, numeric)            | 50000     |
| PerformanceRating | Scale **1 = highest, 5 = lowest** (numeric integer) | 3         |

---

## 🌐 3. External Benchmarking Template

**Required Columns (must match exactly):**
1. JobRole  
2. JobLevel  
3. MarketMedianCTC  

**Column Descriptions with Examples:**

| Column            | Description                                | Example     |
|-------------------|--------------------------------------------|-------------|
| JobRole           | Standardized job role                      | Analyst     |
| JobLevel          | Job grade / level                          | Analyst     |
| MarketMedianCTC   | Median annual benchmark pay (₹, numeric)   | 650000      |

---

## 📂 4. Upload Rules
- 📊 Only Excel **.xlsx** files are accepted.  
- 📑 Templates must be used **as-is** (no custom versions).  
- 🔢 **PerformanceRating** must follow the rule:  
  **1 = highest, 5 = lowest**.  
- ⚠️ If errors occur, download fresh templates and re-enter data.  
- ✅ Once uploaded, headers will be validated before proceeding to insights.

---
"""

def create_howto_pdf_bytes():
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"],
                          fontName=BODY_FONT, fontSize=10, leading=14)
    story = []

    content = get_howto_markdown_full().split("\n\n")
    for block in content:
        if block.strip().startswith("|"):  # Markdown table block
            lines = [ln.strip("|") for ln in block.strip().split("\n") if "|" in ln]
            rows = [ [c.strip() for c in ln.split("|")] for ln in lines ]
            if rows:
                # Strip markdown separators (---)
                if len(rows) > 1 and set("".join(rows[1])) <= set("-: "):
                    rows = [rows[0]] + rows[2:]
                # Zebra style
                tstyle = TableStyle([
                    ("GRID", (0,0), (-1,-1), 0.25, colors.black),
                    ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
                ])
                for r in range(1, len(rows)):
                    if r % 2 == 0:
                        tstyle.add("BACKGROUND", (0,r), (-1,r), TABLE_ZEBRA)
                tbl = Table(rows, repeatRows=1, hAlign="LEFT", colWidths="*")
                tbl.setStyle(tstyle)
                story.append(tbl)
                story.append(Spacer(1,6))
        else:
            story.append(Paragraph(block.replace("\n","<br/>"), body))
            story.append(Spacer(1,6))

    doc.build(story,
              onFirstPage=lambda c,d:(draw_background(c,d), add_page_number(c,d)),
              onLaterPages=lambda c,d:(draw_background(c,d), add_page_number(c,d)))
    return buf.getvalue()
# -----------------------
# App header
# -----------------------
st.markdown(f"""
<div style="padding:18px;border-radius:10px;border:1px solid #ddd;text-align:center">
  <h1 style="margin:0;padding:0">📊 Compensation & Benefits Dashboard</h1>
  <p>Board-ready pay analytics — per-metric filters, exports, and benchmarks.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Step 1: Templates & Guide
# -----------------------
st.header("Step 1 — Templates & Guide")
c1, c2 = st.columns(2)
with c1:
    st.download_button("📥 Internal Compensation Data Template",
                       data=get_employee_template_csv(),
                       file_name="Internal_Compensation_Data_Template.csv")
with c2:
    st.download_button("📥 External Benchmarking Data Template",
                       data=get_benchmark_template_csv(),
                       file_name="External_Benchmarking_Data_Template.csv")

st.download_button("📄 How-to Guide (PDF)",
                   data=create_howto_pdf_bytes(),
                   file_name="How_to_Upload_Guide.pdf")

if not st.checkbox("✅ I downloaded templates + guide"):
    st.stop()

# -----------------------
# Step 2: Upload Data
# -----------------------
st.header("Step 2 — Upload Data")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Internal Compensation CSV/XLSX", type=["csv","xlsx"])
with col2:
    benchmark_file = st.file_uploader("Upload External Benchmarking CSV/XLSX", type=["csv","xlsx"])

if not uploaded_file: st.stop()

def read_input(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")

emp_df = read_input(uploaded_file)
ok, msg = validate_exact_headers(emp_df, EMP_REQUIRED)
if not ok: st.error(msg); st.stop()

bench_df = None
if benchmark_file:
    bench_df = read_input(benchmark_file)
    ok_b, msg_b = validate_exact_headers(bench_df, BENCH_REQUIRED)
    if not ok_b: st.error(msg_b); st.stop()
# -----------------------
# Preview
# -----------------------
st.subheader("Preview Data")
st.dataframe(emp_df.head(10))
if bench_df is not None:
    st.write("Benchmark Preview:")
    st.dataframe(bench_df.head(10))

# -----------------------
# Filters per-metric
# -----------------------
def metric_filters_ui(df, prefix=""):
    st.markdown("**Filters (for this metric only):**")
    c1, c2, c3 = st.columns(3)
    with c1:
        dept = st.selectbox(f"{prefix}Department", ["All"]+sorted(df["Department"].dropna().unique()), key=f"{prefix}_dept")
    with c2:
        roles = sorted(df[df["Department"]==dept]["JobRole"].unique()) if dept!="All" else sorted(df["JobRole"].unique())
        sel_roles = st.multiselect(f"{prefix}Job Role", roles, key=f"{prefix}_roles")
    with c3:
        levels = sorted(df["JobLevel"].dropna().unique())
        sel_levels = st.multiselect(f"{prefix}Job Level", levels, key=f"{prefix}_levels")
    out = df.copy()
    if dept!="All": out=out[out["Department"]==dept]
    if sel_roles: out=out[out["JobRole"].isin(sel_roles)]
    if sel_levels: out=out[out["JobLevel"].isin(sel_levels)]
    return out

# -----------------------
# Quartile categorizer
# -----------------------
def make_quartile_categorizer(series):
    q1,q2,q3 = series.quantile([.25,.5,.75]).tolist()
    iqr=q3-q1; low=q1-1.5*iqr; high=q3+1.5*iqr
    def cat(x):
        if pd.isna(x): return "NA"
        if x<low: return "Outlier"
        if x<=q1: return "Q1"
        if x<=q2: return "Q2"
        if x<=q3: return "Q3"
        if x<=high: return "Q4"
        return "Outlier"
    return cat

# -----------------------
# Metrics storage
# -----------------------
sections=[]; images_for_download=[]

# -----------------------
# Metric A: Avg CTC by Job Level
# -----------------------
st.subheader("🏷️ Average CTC by Job Level")
dfA=metric_filters_ui(emp_df,"A_")
avg=dfA.groupby("JobLevel")["CTC"].mean().reset_index()
avg["AvgCTC_Lakhs"]=avg["CTC"].apply(readable_lakhs_number)
st.dataframe(avg[["JobLevel","AvgCTC_Lakhs"]])
figA=px.bar(avg,x="JobLevel",y="CTC",color="JobLevel",color_discrete_sequence=PALETTE)
assetA=save_plotly_asset(figA,safe_filename("avg_ctc"))
st.plotly_chart(figA)
sections.append(("Average CTC by Job Level","Average CTC across job levels.",avg,assetA))
images_for_download.append({"title":"Average CTC by Job Level","asset":assetA})

# -----------------------
# Metric B: Median CTC by Job Level
# -----------------------
st.subheader("📏 Median CTC by Job Level")
dfB=metric_filters_ui(emp_df,"B_")
med=dfB.groupby("JobLevel")["CTC"].median().reset_index()
med["MedianCTC_Lakhs"]=med["CTC"].apply(readable_lakhs_number)
st.dataframe(med[["JobLevel","MedianCTC_Lakhs"]])
figB=px.bar(med,x="JobLevel",y="CTC",color="JobLevel",color_discrete_sequence=PALETTE)
assetB=save_plotly_asset(figB,safe_filename("median_ctc"))
st.plotly_chart(figB)
sections.append(("Median CTC by Job Level","Median CTC across job levels.",med,assetB))
images_for_download.append({"title":"Median CTC by Job Level","asset":assetB})

# -----------------------
# Metric C: Quartile Placement
# -----------------------
st.subheader("📊 Quartile Placement by Job Level")
dfC=metric_filters_ui(emp_df,"C_")
qcat=make_quartile_categorizer(dfC["CTC"])
rows=[]
for lvl,g in dfC.groupby("JobLevel"):
    vc=g["CTC"].apply(qcat).value_counts()
    rows.append({"JobLevel":lvl,"Count":len(g),
                 "Q1":vc.get("Q1",0),"Q2":vc.get("Q2",0),
                 "Q3":vc.get("Q3",0),"Q4":vc.get("Q4",0),
                 "Outlier":vc.get("Outlier",0)})
quart_tbl=pd.DataFrame(rows)
if not quart_tbl.empty:
    totals=quart_tbl.drop(columns=["JobLevel"]).sum()
    total_row={"JobLevel":"Total",**totals.to_dict()}
    quart_tbl=pd.concat([quart_tbl,pd.DataFrame([total_row])],ignore_index=True)
st.dataframe(quart_tbl)
figC=px.violin(dfC,x="JobLevel",y="CTC",color="JobLevel",box=True,points="all",color_discrete_sequence=PALETTE)
assetC=save_plotly_asset(figC,safe_filename("quartile"))
st.plotly_chart(figC)
sections.append(("Quartile Placement","Quartile counts + distribution.",quart_tbl,assetC))
images_for_download.append({"title":"Quartile Placement","asset":assetC})

# -----------------------
# Metric D: Bonus % by Job Level
# -----------------------
st.subheader("🎁 Avg Bonus % by Job Level")
dfD=metric_filters_ui(emp_df,"D_")
dfD["BonusPct"]=np.where(dfD["CTC"]>0,(dfD["Bonus"]/dfD["CTC"])*100,np.nan)
bonus=dfD.groupby("JobLevel")["BonusPct"].mean().reset_index()
st.dataframe(bonus)
figD=px.bar(bonus,x="JobLevel",y="BonusPct",color="JobLevel",color_discrete_sequence=PALETTE)
assetD=save_plotly_asset(figD,safe_filename("bonus_pct"))
st.plotly_chart(figD)
sections.append(("Avg Bonus % of CTC","Average bonus percent by level.",bonus,assetD))
images_for_download.append({"title":"Avg Bonus % of CTC","asset":assetD})
# -----------------------
# Metric E: Company vs Market
# -----------------------
if bench_df is not None:
    st.subheader("📉 Company vs Market (Median CTC)")
    dfE=metric_filters_ui(emp_df,"E_")
    comp=dfE.groupby("JobLevel")["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
    bench=bench_df.groupby("JobLevel")["MarketMedianCTC"].median().reset_index()
    compare=pd.merge(comp,bench,on="JobLevel",how="outer")
    compare["Gap%"]=np.where(compare["MarketMedianCTC"]>0,(compare["CompanyMedian"]-compare["MarketMedianCTC"])/compare["MarketMedianCTC"]*100,np.nan).round(2)
    st.dataframe(compare)
    figE=go.Figure()
    figE.add_trace(go.Bar(x=compare["JobLevel"],y=compare["CompanyMedian"],name="Company"))
    figE.add_trace(go.Scatter(x=compare["JobLevel"],y=compare["MarketMedianCTC"],name="Market",mode="lines+markers"))
    assetE=save_plotly_asset(figE,safe_filename("cmp_vs_market"))
    st.plotly_chart(figE)
    sections.append(("Company vs Market","Company vs Market medians.",compare,assetE))
    images_for_download.append({"title":"Company vs Market","asset":assetE})

# -----------------------
# Metric F: Avg CTC by Gender & Job Level
# -----------------------
st.subheader("👫 Avg CTC — Gender x Job Level")
dfF=metric_filters_ui(emp_df,"F_")
g=dfF.groupby(["JobLevel","Gender"])["CTC"].mean().reset_index()
g["Lakhs"]=g["CTC"].apply(readable_lakhs_number)
st.dataframe(g.pivot(index="JobLevel",columns="Gender",values="Lakhs").reset_index())
figF=px.bar(g,x="JobLevel",y="CTC",color="Gender",barmode="group",color_discrete_sequence=PALETTE)
assetF=save_plotly_asset(figF,safe_filename("gender_ctc"))
st.plotly_chart(figF)
sections.append(("Avg CTC by Gender & Job Level","Avg CTC split by gender.",g,assetF))
images_for_download.append({"title":"Avg CTC by Gender & Job Level","asset":assetF})

# -----------------------
# Metric G: Avg CTC by Rating & Job Level
# -----------------------
st.subheader("⭐ Avg CTC — Rating x Job Level")
dfG=metric_filters_ui(emp_df,"G_")
r=dfG.groupby(["JobLevel","PerformanceRating"])["CTC"].mean().reset_index()
r["Lakhs"]=r["CTC"].apply(readable_lakhs_number)
st.dataframe(r.pivot(index="JobLevel",columns="PerformanceRating",values="Lakhs").reset_index())
figG=px.bar(r,x="JobLevel",y="CTC",color="PerformanceRating",barmode="group",color_discrete_sequence=PALETTE)
assetG=save_plotly_asset(figG,safe_filename("rating_ctc"))
st.plotly_chart(figG)
sections.append(("Avg CTC by Rating & Job Level","Avg CTC split by rating.",r,assetG))
images_for_download.append({"title":"Avg CTC by Rating & Job Level","asset":assetG})

# -----------------------
# Compiled Report (with PDF bookmarks / clickable outline)
# -----------------------
from reportlab.platypus import Flowable

class PDFBookmark(Flowable):
    """
    Small Flowable to insert a bookmark/outline entry at the current page.
    Helps PDF viewers present a clickable outline pane.
    """
    def __init__(self, name, title):
        super().__init__()
        self.name = name
        self.title = title

    def wrap(self, availWidth, availHeight):
        return (0, 0)

    def draw(self):
        try:
            self.canv.bookmarkPage(self.name)
            self.canv.addOutlineEntry(self.title, self.name, level=0, closed=False)
        except Exception:
            pass  # fail quietly

st.header("📥 Download Reports")
st.write("Choose metrics to include in the compiled PDF:")

# dynamic checkbox set
kpi_check = {title: st.checkbox(title, key=f"chk_{i}") for i, (title, _, _, _) in enumerate(sections)}

if st.button("🧾 Compile Selected Report"):
    selected_titles = [t for t, v in kpi_check.items() if v]
    if not selected_titles:
        st.warning("Select at least one metric to include.")
    else:
        selected_sections = [s for s in sections if s[0] in selected_titles]

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=18*mm, leftMargin=18*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        h1, h2, normal = styles["Title"], styles["Heading2"], styles["Normal"]
        body = ParagraphStyle("body", parent=normal, fontName=BODY_FONT, fontSize=10, leading=13)

        story = []
        # Cover
        story.append(Paragraph("Compensation & Benefits — Compiled Report", h1))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}", normal))
        story.append(PageBreak())

        # TOC
        story.append(Paragraph("Table of Contents", h2))
        for idx, (title, _, _, _) in enumerate(selected_sections, 1):
            story.append(Paragraph(f"{idx}. {title}", normal))
        story.append(PageBreak())

        # Sections
        for title, desc, tbl, asset in selected_sections:
            bname = sanitize_anchor(title)
            story.append(PDFBookmark(bname, title))
            story.append(Paragraph(title, h2))
            story.append(Spacer(1, 6))
            if desc:
                story.append(Paragraph(desc, body))
                story.append(Spacer(1, 6))
            if tbl is not None and not tbl.empty:
                try:
                    data = [list(tbl.columns)] + tbl.fillna("").values.tolist()
                    tstyle = TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),
                                         ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)])
                    for r in range(1, len(data)):
                        if r % 2 == 0:
                            tstyle.add("BACKGROUND", (0,r), (-1,r), TABLE_ZEBRA)
                    t = Table(data, repeatRows=1, hAlign="LEFT")
                    t.setStyle(tstyle)
                    story.append(t)
                    story.append(Spacer(1, 8))
                except Exception as e:
                    story.append(Paragraph(f"Unable to render table: {e}", body))
            if asset:
                if asset.get("png") and os.path.exists(asset["png"]):
                    try:
                        story.append(RLImage(asset["png"], width=170*mm, height=90*mm))
                        story.append(Spacer(1, 6))
                    except Exception:
                        story.append(Paragraph(f"Image exists but couldn't embed: {os.path.basename(asset['png'])}", body))
                elif asset.get("html") and os.path.exists(asset["html"]):
                    story.append(Paragraph(f"Interactive chart saved as HTML: {os.path.basename(asset['html'])}", body))
            story.append(PageBreak())

        # Consolidated Conclusions
        story.append(Paragraph("Consolidated Conclusions (Actionable)", h2))
        conc_rows = [["Metric / JobLevel", "Actionable Insight"]]
        if any(s[0] == "Company vs Market" for s in selected_sections):
            try:
                comp_tbl = next(s for s in selected_sections if s[0] == "Company vs Market")[2]
                if comp_tbl is not None and "Gap %" in comp_tbl.columns:
                    for _, r in comp_tbl.iterrows():
                        jl, gap = r.get("JobLevel", "Unknown"), r.get("Gap %", None)
                        if pd.notna(gap):
                            if gap < -5:
                                conc_rows.append([jl, f"⚠️ {gap}% behind market — consider repricing."])
                            elif gap < 0:
                                conc_rows.append([jl, f"🔍 {gap}% behind market — review."])
                            elif gap > 5:
                                conc_rows.append([jl, f"✅ {gap}% ahead of market — monitor retention."])
                            else:
                                conc_rows.append([jl, f"{gap}% near market — stable."])
            except StopIteration:
                pass
        for title, _, _, _ in selected_sections:
            if title != "Company vs Market":
                conc_rows.append([title, "Review chart/table for insights."])
        try:
            conc_tbl = Table(conc_rows, repeatRows=1, hAlign="LEFT")
            conc_tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),
                                          ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
            story.append(conc_tbl)
        except Exception:
            story.append(Paragraph("Unable to render consolidated conclusions.", body))

        doc.build(story,
                  onFirstPage=lambda c,d:(draw_background(c,d), add_page_number(c,d)),
                  onLaterPages=lambda c,d:(draw_background(c,d), add_page_number(c,d)))
        st.download_button("⬇️ Download Compiled PDF", buf.getvalue(),
                           file_name="cb_dashboard_compiled.pdf", mime="application/pdf")

# -----------------------
# Quick image downloads
# -----------------------
st.subheader("📸 Quick Chart Downloads")
for item in images_for_download:
    title, asset = item.get("title","chart"), item.get("asset",{})
    if asset.get("png") and os.path.exists(asset["png"]):
        with open(asset["png"], "rb") as f:
            st.download_button(f"⬇️ {title} (PNG)", f.read(), file_name=os.path.basename(asset["png"]), mime="image/png")
    elif asset.get("html") and os.path.exists(asset["html"]):
        with open(asset["html"], "rb") as f:
            st.download_button(f"⬇️ {title} (HTML)", f.read(), file_name=os.path.basename(asset["html"]), mime="text/html")

# -----------------------
# Wrap
# -----------------------
st.success("Dashboard loaded ✅ Use per-metric filters and export options. Compiled report supports PDF bookmarks for navigation.")