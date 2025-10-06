# ======================================
# cb_dashboard.py — Compensation & Benefits Dashboard
# Version: 4.6 (QF-8 Stable Polished)
# Last Updated: 2025-10-05
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from io import BytesIO
from datetime import datetime
import os

# ReportLab
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage, Flowable
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
# App Header
# -----------------------
st.markdown("""
<div style="padding:20px;border-radius:12px;border:1px solid #ccc;text-align:center;
background:linear-gradient(180deg,#0E1117 0%,#1E293B 100%);color:white;">
  <h1 style="margin:0;padding:0;font-size:32px;color:#F9FAFB;">📊 Compensation & Benefits Dashboard</h1>
  <p style="font-size:15px;margin-top:6px;color:#D1D5DB;">
    Board-ready pay analytics — per-metric filters, exports, and benchmarks.
  </p>
  <div style="display:inline-block;margin-top:10px;background-color:#FFF3CD;
  color:#856404;padding:6px 14px;border-radius:8px;font-size:13px;font-weight:600;
  border:1px solid #FFECB5;">⚠️ Session resets if idle >5 mins or reloaded. Download PDFs to save results.
  </div>
</div>
""", unsafe_allow_html=True)

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
TABLE_ZEBRA = colors.HexColor("#F7F7F7")
PALETTE = px.colors.qualitative.Vivid
HEADER_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"
AXIS_TITLE_SIZE = 12
AXIS_TICK_SIZE = 11
LEGEND_FONT_SIZE = 12
FALLBACK_PAPER_BG_DARK = "#0b1220"
FALLBACK_PAPER_BG_LIGHT = "#FFFFFF"

def sanitize_anchor(title: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")

def validate_exact_headers(df_or_cols, required_cols):
    cols = list(df_or_cols.columns) if hasattr(df_or_cols, "columns") else list(df_or_cols)
    ok = cols == required_cols
    return (ok, "OK" if ok else f"Header mismatch. Expected {required_cols}, found {cols}")

def readable_lakhs_number(x):
    try: return round(float(x)/1e5,2)
    except: return None
def _detect_theme(theme_arg="auto"):
    try:
        base = st.get_option("theme.base").lower()
        return base if base in ("dark","light") else "dark"
    except: return "dark"

def apply_chart_style(
    fig,
    title: str = None,
    x_title: str = "JobLevel",
    y_title: str = "",
    theme: str = "auto",
    legend_below: bool = True,
    showlegend: bool | None = None
):
    """
    v4.7 — Clean title handling + legend positioning fix
    ✅ Removes 'undefined'
    ✅ Centers padding & font harmonization
    ✅ Legend auto-offsets for overlap fix
    """
    theme = _detect_theme(theme)
    is_dark = theme == "dark"
    text_color = "#FFF" if is_dark else "#000"
    paper_bg = FALLBACK_PAPER_BG_DARK if is_dark else FALLBACK_PAPER_BG_LIGHT
    grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)"
    legend_bg = "rgba(255,255,255,0.03)" if is_dark else "rgba(0,0,0,0.03)"

    if showlegend is None:
        showlegend = len(fig.data) > 1

    # Dynamic legend offset
    legend_y = -0.35 if legend_below else 0.98

    fig.update_layout(
        title=None if not title else dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=16, color=text_color)
        ),
        font=dict(family=BODY_FONT, color=text_color, size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor=paper_bg,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT_SIZE, color=text_color),
            bgcolor=legend_bg,
            bordercolor="rgba(0,0,0,0)"
        ),
        margin=dict(t=50, l=60, r=40, b=120),
        height=440,
    )

    fig.update_xaxes(
        title_text=x_title,
        tickangle=-35,
        tickfont=dict(size=11, color=text_color),
        automargin=True,
        showgrid=False,
    )
    fig.update_yaxes(
        title_text=y_title,
        gridcolor=grid_color,
        tickfont=dict(size=11, color=text_color),
        automargin=True,
    )
    return fig
st.header("Step 1 — Templates & Guide")
c1,c2=st.columns(2)
def get_employee_template_csv(): 
    return pd.DataFrame([["E1001","Male","Finance","Analyst","Analyst",600000,50000,3]],
        columns=EMP_REQUIRED).to_csv(index=False)
def get_benchmark_template_csv(): 
    return pd.DataFrame([["Analyst","Analyst",650000]],columns=BENCH_REQUIRED).to_csv(index=False)
c1.download_button("📥 Internal Template",get_employee_template_csv(),"Internal_Template.csv")
c2.download_button("📥 Benchmark Template",get_benchmark_template_csv(),"Benchmark_Template.csv")
if not st.checkbox("✅ Templates downloaded"): st.stop()

st.header("Step 2 — Upload Data")
col1,col2=st.columns(2)
up=col1.file_uploader("Upload Internal Compensation Data",["csv","xlsx"])
bm=col2.file_uploader("Upload Benchmark Data (optional)",["csv","xlsx"])
if not up: st.stop()

read=lambda f: pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f,engine="openpyxl")
emp_df=read(up); ok,msg=validate_exact_headers(emp_df,EMP_REQUIRED)
if not ok: st.error(msg); st.stop()
bench_df=None
if bm:
    bench_df=read(bm); ok,msg=validate_exact_headers(bench_df,BENCH_REQUIRED)
    if not ok: st.error(msg); st.stop()
def _safe_numeric(df,col,fill_zero=False):
    df=df.copy(); df[col]=pd.to_numeric(df[col],errors="coerce")
    if fill_zero: df[col]=df[col].fillna(0)
    return df

def _ensure_joblevel_order(df,col="JobLevel"):
    order=["Analyst","Assistant Manager","Manager","Senior Manager",
           "Associate Partner","Director","Executive","Senior Executive"]
    if col in df.columns:
        df=df.copy(); df[col]=pd.Categorical(df[col],categories=order,ordered=True)
    return df
#===========
# Metric 1
#===========
def average_ctc_by_joblevel(df, job_col="JobLevel", ctc_col="CTC"):
    df=_safe_numeric(df,ctc_col); df=_ensure_joblevel_order(df,job_col)
    agg=df.groupby(job_col,observed=True)[ctc_col].agg(["mean","sum"]).reset_index()
    agg["Total CTC (₹ Cr.)"]=(agg["sum"]/1e7).round(2)
    agg["Avg CTC (₹ Lakhs)"]=(agg["mean"]/1e5).round(2)
    agg=agg[[job_col,"Total CTC (₹ Cr.)","Avg CTC (₹ Lakhs)"]]
    fig=px.bar(agg,x=job_col,y="Avg CTC (₹ Lakhs)",color=job_col,
               text="Avg CTC (₹ Lakhs)",color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="outside")
    fig=apply_chart_style(fig,title=" ", showlegend=False)
    return agg, fig

#===========
# Metric 2
#===========

def median_ctc_by_joblevel(df, job_col="JobLevel", ctc_col="CTC"):
    df=_safe_numeric(df,ctc_col); df=_ensure_joblevel_order(df,job_col)
    agg=df.groupby(job_col,observed=True)[ctc_col].agg(["median","sum"]).reset_index()
    agg["Total CTC (₹ Cr.)"]=(agg["sum"]/1e7).round(2)
    agg["Median CTC (₹ Lakhs)"]=(agg["median"]/1e5).round(2)
    agg=agg[[job_col,"Total CTC (₹ Cr.)","Median CTC (₹ Lakhs)"]]
    fig=px.bar(agg,x=job_col,y="Median CTC (₹ Lakhs)",color=job_col,
               text="Median CTC (₹ Lakhs)",color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="outside")
    fig=apply_chart_style(fig,title=" ", showlegend=False)
    return agg, fig
#===========
# Metric 3
#===========
# --- FIX C (v4.6.1 Stable) ---
#===========
def quartile_distribution(df, ctc_col="CTC", job_col="JobLevel"):
    """
    Quartile Distribution (Fixed):
    ✅ Uses employee counts instead of % share.
    ✅ Adds totals & grand total row.
    ✅ Donut reflects overall quartile distribution.
    ✅ Handles missing/auto-generated column name bug.
    """
    df = _safe_numeric(df, ctc_col)
    df = _ensure_joblevel_order(df, job_col)
    df = df.copy()

    # Step 1 — Assign quartiles dynamically
    try:
        df["Quartile"] = pd.qcut(df[ctc_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    except Exception:
        df["Quartile"] = pd.cut(df[ctc_col], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])

    # Step 2 — Employee count per quartile and job level
    quartile_counts = pd.crosstab(df[job_col], df["Quartile"]).reset_index()

    # Step 3 — Add totals per job level
    quartile_counts["Total Employees"] = quartile_counts[["Q1", "Q2", "Q3", "Q4"]].sum(axis=1)

    # Step 4 — Add grand total row
    grand_totals = quartile_counts[["Q1", "Q2", "Q3", "Q4", "Total Employees"]].sum()
    grand_totals[job_col] = "Grand Total"
    quartile_counts = pd.concat([quartile_counts, pd.DataFrame([grand_totals])], ignore_index=True)

    # Step 5 — Calculate grand total % distribution (for donut)
    total_employees = quartile_counts.loc[quartile_counts[job_col] == "Grand Total", "Total Employees"].values[0]

    donut_df = (
        quartile_counts[["Q1", "Q2", "Q3", "Q4"]]
        .iloc[-1]
        .drop("Total Employees", errors="ignore")
        .reset_index()
        .rename(columns={"index": "Quartile", 0: "Count"})
    )

    # 🔧 Patch: Ensure correct column name exists
    if "Count" not in donut_df.columns:
        donut_df.columns = ["Quartile", "Count"]

    donut_df["Percent"] = (donut_df["Count"].astype(float) / total_employees * 100).round(1)

    # Step 6 — Donut chart: overall quartile distribution (company-wide)
    fig = go.Figure(go.Pie(
    labels=donut_df["Quartile"],
    values=donut_df["Count"],
    hole=0.5,
    textinfo="label+percent",
    insidetextorientation="radial",
    marker=dict(colors=PALETTE[:4], line=dict(color="#0E1117", width=2))
))
    fig=apply_chart_style(fig,title=" ", showlegend=False)

    # Step 7 — Final Output Table
    quartile_counts = quartile_counts.fillna(0)
    quartile_counts = quartile_counts.astype({c: int for c in ["Q1", "Q2", "Q3", "Q4", "Total Employees"] if c in quartile_counts.columns})
    return quartile_counts, fig

#===========
# Metric 4
#===========

def company_vs_market(df_company,df_market,job_col="JobLevel",
                      company_col="CompanyMedian",market_col="MarketMedian"):
    left=_ensure_joblevel_order(df_company,job_col)
    right=_ensure_joblevel_order(df_market,job_col)
    merged=pd.merge(left,right,on=job_col,how="inner").dropna()
    merged["Company (₹ L)"]=(merged[company_col]/1e5).round(2)
    merged["Market (₹ L)"]=(merged[market_col]/1e5).round(2)
    merged["Gap %"]=((merged[company_col]-merged[market_col])/merged[market_col]*100).round(1)
    table=merged[[job_col,"Company (₹ L)","Market (₹ L)","Gap %"]]
    fig=go.Figure([
        go.Bar(x=merged[job_col],y=merged["Company (₹ L)"],name="Company",marker_color="#22D3EE"),
        go.Scatter(x=merged[job_col],y=merged["Market (₹ L)"],name="Market",
                   mode="lines+markers",line=dict(color="#FB7185",width=3))
    ])
    fig=apply_chart_style(fig,title=" ", showlegend=True)
    fig.update_layout(legend=dict(font=dict(size=10)))
    return table,fig

#===========
# Metric 5
#===========

def bonus_pct_by_joblevel(df,job_col="JobLevel",bonus_col="Bonus",ctc_col="CTC"):
    df=_safe_numeric(df,ctc_col); df=_safe_numeric(df,bonus_col); df=_ensure_joblevel_order(df,job_col)
    df["Bonus %"]=np.where(df[ctc_col]>0,(df[bonus_col]/df[ctc_col])*100,np.nan)
    agg=df.groupby(job_col,observed=True)["Bonus %"].mean().reset_index().round(2)
    fig=px.bar(agg,x=job_col,y="Bonus %",color=job_col,text="Bonus %",color_discrete_sequence=PALETTE)
    fig=apply_chart_style(fig,title=" ", showlegend=False)
    return agg,fig

#===========
# Metric 6
#===========

def average_ctc_by_gender_joblevel(df,job_col="JobLevel",gender_col="Gender",ctc_col="CTC"):
    df=_safe_numeric(df,ctc_col); df=df.dropna(subset=[ctc_col]); df=_ensure_joblevel_order(df,job_col)
    agg=df.groupby([job_col,gender_col],observed=True)[ctc_col].mean().reset_index()
    agg["CTC_L"]=(agg[ctc_col]/1e5).round(2)
    pivot=agg.pivot(index=job_col,columns=gender_col,values="CTC_L").fillna(np.nan)
    pivot["Gap %"]=np.where(pivot.get("Female",np.nan)>0,
                            ((pivot.get("Male",0)-pivot.get("Female",0))/pivot.get("Female",0)*100).round(1),np.nan)
    pivot=pivot.reset_index().rename(columns={"Male":"Avg CTC (M)","Female":"Avg CTC (F)"})
    fig=px.bar(agg,x=job_col,y="CTC_L",color=gender_col,barmode="group",color_discrete_sequence=PALETTE)
    fig=apply_chart_style(fig,title=" ", showlegend=True)
    return pivot,fig

# ===========
# Metric 7 (v4.7.3 Final Polish)
# ===========
def average_ctc_by_rating_joblevel(df, job_col="JobLevel", rating_col="Rating", ctc_col="CTC"):
    df = _safe_numeric(df, ctc_col)
    df = _ensure_joblevel_order(df, job_col)
    agg = df.groupby([job_col, rating_col], observed=True)[ctc_col].mean().reset_index()
    agg["CTC_L"] = (agg[ctc_col] / 1e5).round(2)
    pivot = agg.pivot(index=job_col, columns=rating_col, values="CTC_L").round(2).reset_index()

    # ✅ Use a bright-friendly palette (replaces dark Blues)
    light_blues = ["#A7C7E7", "#7FB3E4", "#5397D3", "#3C77B9", "#1E5AA8"]

    fig = px.bar(
        agg,
        x=job_col,
        y="CTC_L",
        color=rating_col,
        barmode="stack",
        color_discrete_sequence=light_blues
    )

    # ✅ Proper styling — keep titles invisible in app
    fig = apply_chart_style(fig, title=" ", showlegend=True)
    return pivot, fig
# ===========================
# Render Metrics + Tables (v4.7 Final Stable Polish)
# ===========================

sections = []
images_for_download = []
# ==========================
# Helper: Chart Image Saver (v4.7.4 Stable + Smart Contrast Fix)
# ========================
def save_chart_image(title, fig):
    """
    Saves Plotly chart as high-quality PNG inside temp_charts_cb/
    ✅ Handles errors silently
    ✅ Ensures consistent naming & scaling
    ✅ Smart contrast-aware background for PDF exports
    """
    try:
        img_path = os.path.join(TMP_DIR, f"{sanitize_anchor(title)}.png")

        # ⚡ Smart contrast fix:
        # Keep text/axis labels visible regardless of theme
        fig.update_layout(
            paper_bgcolor="#F4F4F4",
            plot_bgcolor="#F4F4F4",
            font=dict(color="#000"),
            xaxis=dict(color="#000", title_font=dict(color="#000"), tickfont=dict(color="#000")),
            yaxis=dict(color="#000", title_font=dict(color="#000"), tickfont=dict(color="#000")),
            legend=dict(font=dict(color="#000"))
        )

        # 🧩 Ensure marker visibility in light mode (esp. stacked bars)
        for trace in fig.data:
            if hasattr(trace, "marker"):
                trace.marker.line = dict(width=0.5, color="#DDD")

        fig.write_image(img_path, width=1200, height=700, scale=2)
        return img_path

    except Exception as e:
        st.warning(f"⚠️ Could not save image for {title}: {e}")
        return None
# --- Main Metric Group (A → D) ---
metrics = [
    ("🏷️ Average CTC by Job Level", average_ctc_by_joblevel,
     "Average pay by level."),
    ("📏 Median CTC by Job Level", median_ctc_by_joblevel,
     "Median pay across levels."),
    ("📊 Quartile Distribution (Share of Employees)", quartile_distribution,
     "Distribution of employees by quartile across levels."),
    ("🎁 Bonus % of CTC by Job Level", bonus_pct_by_joblevel,
     "Average bonus percentage by level.")
]

for title, func, desc in metrics:
    st.subheader(title)
    table, fig = func(emp_df)
    st.dataframe(table, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)

    # Save chart image (A–D)
    img_path = save_chart_image(title, fig)
    sections.append((title, desc, table, {"png": {"path": img_path}}))

# --- Company vs Market Median (E) ---
if bench_df is not None:
    st.subheader("📉 Company vs Market Median")
    df_company = (
        emp_df.groupby("JobLevel", observed=True)["CTC"]
        .median().reset_index().rename(columns={"CTC": "CompanyMedian"})
    )
    df_market = (
        bench_df.groupby("JobLevel", observed=True)["MarketMedianCTC"]
        .median().reset_index().rename(columns={"MarketMedianCTC": "MarketMedian"})
    )
    tableE, figE = company_vs_market(df_company, df_market)
    figE = apply_chart_style(figE, title=" ", showlegend=True)
    figE.update_layout(
        legend=dict(
            orientation="v",
            xanchor="right",
            x=1,
            yanchor="top",
            y=1,
            font=dict(size=9)
        )
    )
    st.dataframe(tableE, use_container_width=True)
    st.plotly_chart(figE, use_container_width=True)
    img_path = save_chart_image("Company vs Market Median", figE)
    sections.append(("Company vs Market Median", "Internal vs market comparison.", tableE, {"png": {"path": img_path}}))
else:
    st.info("ℹ️ Upload benchmark data to view market comparison.")

# ------------------------------------------------------------
# Gender & Rating Differentiation (F & G)
# ------------------------------------------------------------
emp_df = emp_df.rename(columns={"PerformanceRating": "Rating"})

last_metrics = [
    ("👫 Average CTC by Gender & Job Level", average_ctc_by_gender_joblevel,
     "Gender pay differentiation across levels."),
    ("⭐ Average CTC by Performance Rating & Job Level", average_ctc_by_rating_joblevel,
     "Pay differentiation by performance rating.")
]

for title, func, desc in last_metrics:
    st.subheader(title)
    table, fig = func(emp_df)
    fig = apply_chart_style(fig, title=" ", showlegend=True)
    st.dataframe(table, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)

    # Save chart image (F–G)
    img_path = save_chart_image(title, fig)
    sections.append((title, desc, table, {"png": {"path": img_path}}))
#----------------------
# PDF Bookmark Helper
# -----------------------
class PDFBookmark(Flowable):
    def __init__(self, name, title):
        super().__init__()
        self.name, self.title = name, title

    def wrap(self, availWidth, availHeight):
        return (0, 0)

    def draw(self):
        try:
            self.canv.bookmarkPage(self.name)
            self.canv.addOutlineEntry(self.title, self.name, level=0, closed=False)
        except Exception:
            pass
#======================
# Download Reports (PDF)
#======================
st.header("📥 Download Reports")
st.write("Choose metrics to include in the compiled PDF:")

# Checkbox selection for all available metrics
kpi_check = {
    title: st.checkbox(title, key=f"chk_{i}")
    for i, (title, _, _, _) in enumerate(sections)
}

if st.button("🧾 Compile Selected Report"):
    selected = [s for s in sections if kpi_check.get(s[0])]
    if not selected:
        st.warning("Select at least one metric to include.")
    else:
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=18*mm,
            leftMargin=18*mm,
            topMargin=20*mm,
            bottomMargin=20*mm,
        )

        styles = getSampleStyleSheet()
        body = ParagraphStyle(
            "body",
            parent=styles["Normal"],
            fontName=BODY_FONT,
            fontSize=10,
            leading=13,
        )

        story = []

        # === Cover Page ===
        story.append(
            Paragraph(
                "<para align=center><font size=26 color='#4B0082'><b>Compensation & Benefits Report</b></font></para>",
                body,
            )
        )
        story.append(Spacer(1, 18))
        story.append(
            Paragraph(
                f"<para align=center>Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}</para>",
                body,
            )
        )
        story.append(PageBreak())

        # === Table of Contents ===
        story.append(Paragraph("<b>Table of Contents</b>", styles["Heading2"]))
        toc_data = [[f"{i}.", title] for i, (title, _, _, _) in enumerate(selected, 1)]
        toc_table = Table(toc_data, colWidths=[20 * mm, 150 * mm])
        toc_style = TableStyle(
            [
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, -1), BODY_FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
        toc_table.setStyle(toc_style)
        story.append(toc_table)
        story.append(PageBreak())

        # === Section Pages ===
        for title, desc, tbl, asset in selected:
            story.append(PDFBookmark(sanitize_anchor(title), title))
            story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
            if desc:
                story.append(Paragraph(desc, body))
                story.append(Spacer(1, 6))

            # --- Table rendering ---
            if tbl is not None and not tbl.empty:
                tbl = tbl.astype(str).fillna("")
                data = [list(tbl.columns)] + tbl.values.tolist()
                col_width = (A4[0] - 40) / len(tbl.columns)
                t = Table(data, colWidths=[col_width] * len(tbl.columns), repeatRows=1)

                t_style = TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ]
                )
                for r in range(1, len(data)):
                    if r % 2 == 0:
                        t_style.add(
                            "BACKGROUND", (0, r), (-1, r), TABLE_ZEBRA
                        )

                t.setStyle(t_style)
                story.append(t)
                story.append(Spacer(1, 8))

            # --- Add chart image if available ---
            try:
                img_path = None
                if isinstance(asset, dict):
                    img_path = (
                        asset.get("png", {}).get("path")
                        if isinstance(asset.get("png"), dict)
                        else None
                    )

                if img_path and os.path.exists(img_path):
                    story.append(Spacer(1, 6))
                    story.append(RLImage(img_path, width=160 * mm, height=90 * mm))
                    story.append(Spacer(1, 10))
            except Exception as e:
                st.warning(f"⚠️ Could not embed chart for {title}: {e}")

            story.append(Paragraph("<i>Insight:</i> Review trends across levels.", body))
            story.append(PageBreak())

        # === Build PDF ===
        doc.build(story)

        # === Download Button ===
        st.download_button(
            "⬇️ Download Compiled PDF",
            buf.getvalue(),
            file_name="cb_dashboard_compiled.pdf",
            mime="application/pdf",
        )
# -----------------------
# Quick Chart Downloads (Stable v4.7)
# -----------------------
st.subheader("📸 Quick Chart Downloads (v4.7)")
for s in sections:
    title = s[0]
    asset = s[3] if len(s) > 3 else {}
    png_path = asset.get("png", {}).get("path") if isinstance(asset.get("png"), dict) else None

    if png_path and os.path.exists(png_path):
        with open(png_path, "rb") as f:
            st.download_button(
                label=f"⬇️ {title} (PNG)",
                data=f.read(),
                file_name=os.path.basename(png_path),
                mime="image/png",
            )

st.success("✅ Dashboard Loaded — All metrics now include images & clean legends.")
#=================
# Chatbot Section
#=================
def run_chatbot_ui():
    st.subheader("💬 C&B Data Chatbot")
    if "messages" not in st.session_state: st.session_state["messages"]=[]
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about CTC, Bonus %, Gender Gap, Market vs Company..."):
        q = prompt.lower(); st.session_state["messages"].append({"role":"user","content":prompt})
        res = "🤔 Try asking about **average CTC**, **bonus %**, **gender gap**, or **market comparison**."

        if "average" in q and "ctc" in q:
            avg=emp_df.groupby("JobLevel")["CTC"].mean().reset_index()
            avg["Lakhs"]=avg["CTC"].apply(readable_lakhs_number)
            res=f"📊 **Average CTC by Level:**\n\n{avg.to_markdown(index=False)}"
            st.plotly_chart(px.bar(avg,x="JobLevel",y="Lakhs",color="JobLevel",title="Average CTC (L)"))

        elif "gender" in q or "pay gap" in q:
            g=emp_df.groupby(["JobLevel","Gender"])["CTC"].mean().reset_index()
            g["Lakhs"]=g["CTC"].apply(readable_lakhs_number)
            res=f"👫 **Gender Pay Gap:**\n\n{g.pivot(index='JobLevel',columns='Gender',values='Lakhs').to_markdown()}"
            st.plotly_chart(px.bar(g,x="JobLevel",y="Lakhs",color="Gender",barmode="group"))

        elif "bonus" in q:
            dfB=emp_df.assign(**{"Bonus %":(emp_df["Bonus"]/emp_df["CTC"]*100).round(2)})
            res=f"🎁 **Bonus % by Level:**\n\n{dfB.groupby('JobLevel')['Bonus %'].mean().round(2).to_markdown()}"
            st.plotly_chart(px.bar(dfB,x="JobLevel",y="Bonus %",color="JobLevel",title="Bonus %"))

        elif "market" in q and bench_df is not None:
            comp=emp_df.groupby("JobLevel")["CTC"].median().reset_index()
            bench=bench_df.groupby("JobLevel")["MarketMedianCTC"].median().reset_index()
            cmp=pd.merge(comp,bench,on="JobLevel",how="inner")
            res=f"📉 **Company vs Market Median:**\n\n{cmp.to_markdown(index=False)}"
            st.plotly_chart(px.line(cmp,x="JobLevel",y=["CTC","MarketMedianCTC"],title="Market Comparison"))

        st.session_state["messages"].append({"role":"assistant","content":res})
        with st.chat_message("assistant"): st.markdown(res)

st.sidebar.subheader("🤖 Chatbot Assistant")
if st.sidebar.checkbox("Enable Chatbot", value=False): run_chatbot_ui()