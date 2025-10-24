# modules/consolidated_module.py — v6.0 | Consolidated HR Deck (Background PDF Builder)
import os
import io
import time
import threading
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px

# ReportLab imports for building the PDF in background
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Consolidated utilities
from utils_consolidated.chart_consolidated_saver import ensure_chart_saved
from utils_consolidated.uploader_consolidated_helper import upload_data

# -------------------------
# Configuration
# -------------------------
TMP_PDF_DIR = "/tmp"
os.makedirs(TMP_PDF_DIR, exist_ok=True)

# Try to register DejaVu for unicode (currency symbols, etc.)
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    FONT_NAME = "DejaVuSans"
except Exception:
    FONT_NAME = "Helvetica"

PRIMARY_COLOR = colors.HexColor("#1E3A8A")
ACCENT_COLOR = colors.HexColor("#2563EB")
HEADER_COLOR = colors.HexColor("#0F172A")
TABLE_HEADER_BG = colors.HexColor("#E5E7EB")
BODY_TEXT = colors.HexColor("#111827")

# -------------------------
# UI Header + small CSS
# -------------------------
st.markdown("""
<div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
  <h2 style="margin:0">📘 Consolidated HR Leadership Deck</h2>
  <p style="margin:4px 0 0 0;">Unified boardroom-ready executive report across all modules (background PDF generation — no freezing).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid*="stFileUploader"], div[data-testid*="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, #1E293B, #0F172A) !important;
    border: 1px solid #1E3A8A !important;
    border-radius: 12px !important;
    color: #E5E7EB !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("### 🧩 Upload All Module Datasets")
st.caption("Upload the same data files used in the individual modules — this deck will consolidate metrics & charts automatically.")

# -------------------------
# Upload inputs (using consolidated uploader helper)
# -------------------------
c1, c2, c3 = st.columns(3)
attr_df = upload_data("📉 Attrition Data", key="con_attr")
comp_df = upload_data("💰 Compensation Data", key="con_comp")
perf_df = upload_data("🏆 Performance Data", key="con_perf")
c4, c5 = st.columns(2)
eng_df = upload_data("💬 Engagement Data", key="con_eng")
work_df = upload_data("🏢 Workforce Data", key="con_work")

# Block until all provided
if not all(df is not None for df in [attr_df, comp_df, perf_df, eng_df, work_df]):
    st.info("📥 Please upload all five datasets to proceed.")
    st.stop()

# -------------------------
# Small Data helpers
# -------------------------
def _round_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float", "int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2

def _safe_mean(series):
    try:
        return float(series.mean())
    except Exception:
        return None

# -------------------------
# Build charts/data blocks (identical payload shape as before)
# -------------------------
def build_module_payloads(attr_df, comp_df, perf_df, eng_df, work_df):
    payloads = []

    # Attrition
    attr_blocks = []
    if "AttritionFlag" in attr_df.columns:
        attr_rate = (attr_df["AttritionFlag"].astype(str).str.lower().isin(["yes","y","1","true"]).mean()) * 100
        avg_tenure = _safe_mean(attr_df.get("TenureMonths", pd.Series(dtype=float)))
        dept = attr_df.groupby("Department", observed=True)["AttritionFlag"].apply(
            lambda x: (x.astype(str).str.lower().isin(["yes","y","1","true"])).mean() * 100
        ).reset_index(name="AttritionRate")
        fig_attr = px.bar(dept, x="Department", y="AttritionRate", color="Department", title="Attrition % by Department")
        attr_blocks = [
            {"title":"Attrition Overview","desc":"Overall attrition metrics","df":pd.DataFrame({"Attrition %":[round(attr_rate,2)], "Avg Tenure (mo)":[round(avg_tenure,2) if avg_tenure else "N/A"]}), "fig": None, "insights":[f"Attrition rate: {attr_rate:.1f}%", f"Avg tenure: {avg_tenure:.1f} mo" if avg_tenure else "N/A"]},
            {"title":"Departmental Attrition","desc":"Attrition % by Department","df":_round_df(dept),"fig":fig_attr,"insights":[]}
        ]
    payloads.append({"module_name":"Attrition","module_desc":"Turnover & tenure trends","data_blocks":attr_blocks})

    # Compensation
    comp_blocks = []
    if "CTC" in comp_df.columns:
        comp_df["CTC"] = pd.to_numeric(comp_df["CTC"], errors="coerce")
        comp_df["Bonus"] = pd.to_numeric(comp_df.get("Bonus", 0), errors="coerce")
        comp_df["BonusPct"] = (comp_df["Bonus"] / comp_df["CTC"].replace(0, None)) * 100
        ctc = comp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index(name="AvgCTC")
        bonus = comp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index(name="AvgBonusPct")
        fig_ctc = px.bar(ctc, x="JobLevel", y="AvgCTC", color="JobLevel", title="Avg CTC by Job Level")
        fig_bonus = px.bar(bonus, x="JobLevel", y="AvgBonusPct", color="JobLevel", title="Bonus % by Job Level")
        comp_blocks = [
            {"title":"CTC by Job Level","desc":"Average internal pay per level","df":_round_df(ctc),"fig":fig_ctc,"insights":[]},
            {"title":"Bonus by Job Level","desc":"Average bonus % per level","df":_round_df(bonus),"fig":fig_bonus,"insights":[]}
        ]
    payloads.append({"module_name":"Compensation","module_desc":"Pay & incentive analytics","data_blocks":comp_blocks})

    # Performance
    perf_blocks = []
    if "PerformanceRating" in perf_df.columns:
        perf_df["PerformanceRating"] = pd.to_numeric(perf_df["PerformanceRating"], errors="coerce")
        job_perf = perf_df.groupby("JobLevel", observed=True)["PerformanceRating"].mean().reset_index(name="AvgRating")
        fig_perf = px.bar(job_perf, x="JobLevel", y="AvgRating", color="JobLevel", title="Avg Performance Rating by Job Level")
        perf_blocks = [{"title":"Performance Summary","desc":"Average rating per job level","df":_round_df(job_perf),"fig":fig_perf,"insights":[f"Overall avg rating: {perf_df['PerformanceRating'].mean():.2f}"]}]
    payloads.append({"module_name":"Performance","module_desc":"Performance distribution & KPIs","data_blocks":perf_blocks})

    # Engagement
    eng_blocks = []
    qcols = [c for c in eng_df.columns if c.upper().startswith("Q")]
    if qcols:
        eng_df[qcols] = eng_df[qcols].apply(pd.to_numeric, errors="coerce")
        eng_df["EngagementIndex"] = eng_df[qcols].mean(axis=1)
        dept_eng = eng_df.groupby("Department", observed=True)["EngagementIndex"].mean().reset_index(name="MeanIndex")
        fig_eng = px.bar(dept_eng, x="Department", y="MeanIndex", color="Department", title="Engagement Index by Department")
        eng_blocks = [
            {"title":"Engagement Overview","desc":"Overall engagement index","df":pd.DataFrame({"Average Index":[eng_df['EngagementIndex'].mean().round(2)]}), "fig": None, "insights":[f"Avg engagement index: {eng_df['EngagementIndex'].mean():.2f}"]},
            {"title":"Departmental Engagement","desc":"Avg engagement by department","df":_round_df(dept_eng),"fig":fig_eng,"insights":[]}
        ]
    payloads.append({"module_name":"Engagement","module_desc":"Survey sentiment & participation","data_blocks":eng_blocks})

    # Workforce
    work_blocks = []
    if "JobLevel" in work_df.columns:
        headcount = work_df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount")
        fig_hc = px.bar(headcount, x="JobLevel", y="Headcount", color="JobLevel", title="Headcount by Job Level")
        gender_split = work_df["Gender"].value_counts(normalize=True).mul(100).reset_index()
        gender_split.columns = ["Gender","Percent"]
        fig_gender = px.pie(gender_split, names="Gender", values="Percent", title="Gender Composition")
        work_blocks = [
            {"title":"Headcount Structure","desc":"Employee count by level","df":headcount,"fig":fig_hc,"insights":[]},
            {"title":"Gender Composition","desc":"Gender % across workforce","df":gender_split,"fig":fig_gender,"insights":[]}
        ]
    payloads.append({"module_name":"Workforce","module_desc":"Structure & diversity insights","data_blocks":work_blocks})

    return payloads

modules_payload = build_module_payloads(attr_df, comp_df, perf_df, eng_df, work_df)

# -------------------------
# Background PDF builder (writes PDF to out_path)
# -------------------------
def _add_footer(canvas, doc):
    canvas.saveState()
    footer_text = "Prepared with ❤️ by People Analytics Project — 2025"
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0] / 2, 15, footer_text)
    canvas.restoreState()

def build_pdf_to_path(report_title: str, modules_payload: list, out_path: str):
    """Synchronous builder that writes file to out_path (used in background)."""
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("Title", fontName=FONT_NAME, fontSize=22, alignment=1, textColor=HEADER_COLOR, leading=26)
        subtitle_style = ParagraphStyle("Subtitle", fontName=FONT_NAME, fontSize=12, alignment=1, textColor=colors.HexColor("#374151"))
        small_grey = ParagraphStyle("SmallGrey", fontName=FONT_NAME, fontSize=9, textColor=colors.HexColor("#6B7280"))
        heading = ParagraphStyle("Heading", fontName=FONT_NAME, fontSize=14, textColor=PRIMARY_COLOR, spaceAfter=6)
        body = ParagraphStyle("Body", fontName=FONT_NAME, fontSize=10, textColor=BODY_TEXT, leading=13)

        story = []

        # Cover
        story.append(Spacer(1, 60))
        story.append(Paragraph(f"<b>{report_title}</b>", title_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph("People Analytics — Leadership Insights Deck", subtitle_style))
        story.append(Spacer(1, 24))
        story.append(Paragraph(f"<font size=9>Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}</font>", small_grey))
        story.append(PageBreak())

        # TOC
        toc_rows = [["#", "Module", "Description", "Page"]]
        for i, m in enumerate(modules_payload, 1):
            toc_rows.append([i, m.get("module_name", ""), m.get("module_desc", ""), str(i + 1)])
        toc_table = Table(toc_rows, colWidths=[25, 110, 240, 35])
        toc_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), TABLE_HEADER_BG),
            ("GRID", (0,0), (-1,-1), 0.25, colors.black),
            ("FONTNAME", (0,0), (-1,0), FONT_NAME),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("ALIGN", (0,0), (-1,-1), "LEFT")
        ]))
        story.append(Paragraph("<b>Table of Contents</b>", heading))
        story.append(Spacer(1, 6))
        story.append(toc_table)
        story.append(PageBreak())

        # Modules
        executive_summary = [["Module", "Key Insights"]]
        for mod in modules_payload:
            module_name = mod.get("module_name", "Module")
            module_desc = mod.get("module_desc", "")
            data_blocks = mod.get("data_blocks", [])

            # Divider page for module
            story.append(Spacer(1, 40))
            story.append(Paragraph(f"<para align=center><font size=16 color='white'><b>{module_name.upper()}</b></font></para>",
                                   ParagraphStyle("Divider", backColor=PRIMARY_COLOR, alignment=1, spaceBefore=20, spaceAfter=20, leading=20)))
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<para align=center><font size=10 color='#6B7280'>{module_desc}</font></para>", small_grey))
            story.append(PageBreak())

            # For each block
            for block in data_blocks:
                title = block.get("title","")
                desc = block.get("desc","")
                df = block.get("df", None)
                fig = block.get("fig", None)
                insights = block.get("insights", [])

                story.append(Paragraph(f"{title}", heading))
                story.append(Paragraph(desc, body))
                story.append(Spacer(1, 8))

                # Table
                if df is not None and not df.empty:
                    df = df.round(2).astype(str)
                    table_data = [list(df.columns)] + df.values.tolist()
                    col_count = len(df.columns)
                    table = Table(table_data, colWidths=[(A4[0] - 40) / col_count] * col_count, repeatRows=1)
                    table.setStyle(TableStyle([
                        ("GRID", (0,0), (-1,-1), 0.25, colors.black),
                        ("BACKGROUND", (0,0), (-1,0), TABLE_HEADER_BG),
                        ("FONTNAME", (0,0), (-1,-1), FONT_NAME),
                        ("FONTSIZE", (0,0), (-1,-1), 9),
                        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                        ("LEFTPADDING", (0,0), (-1,-1), 4),
                        ("RIGHTPADDING", (0,0), (-1,-1), 4),
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 8))

                # Chart: export to image file using ensure_chart_saved (will try kaleido)
                if fig is not None:
                    try:
                        img_path = ensure_chart_saved(title, fig)
                        if img_path and os.path.exists(img_path):
                            story.append(RLImage(img_path, width=170*mm, height=95*mm))
                            story.append(Spacer(1, 8))
                        else:
                            story.append(Paragraph("⚠️ Chart could not be rendered.", body))
                    except Exception as e:
                        story.append(Paragraph(f"⚠️ Chart export error: {e}", body))

                # Insights
                if insights:
                    joined = " • ".join(str(x) for x in insights)
                    story.append(Paragraph(f"<font color='{ACCENT_COLOR}'><i>{joined}</i></font>", body))
                    executive_summary.append([module_name, joined])
                else:
                    executive_summary.append([module_name, "No explicit insights."])

                story.append(PageBreak())

        # Executive summary
        story.append(Paragraph("Executive Summary", heading))
        story.append(Spacer(1, 8))
        summary_table = Table(executive_summary, colWidths=[140, 310])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), TABLE_HEADER_BG),
            ("GRID", (0,0), (-1,-1), 0.25, colors.black),
            ("FONTNAME", (0,0), (-1,-1), FONT_NAME),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(summary_table)

        # Build to buffer then write atomically to out_path
        doc.build(story, onLaterPages=_add_footer)
        pdf_bytes = buf.getvalue()

        # atomic write
        tmp = f"{out_path}.tmp"
        with open(tmp, "wb") as f:
            f.write(pdf_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
        return True, None

    except Exception as exc:
        tb = traceback.format_exc()
        return False, tb

# -------------------------
# Thread management and UI state
# -------------------------
if "con_pdf_status" not in st.session_state:
    st.session_state["con_pdf_status"] = "idle"  # idle | running | ready | error
    st.session_state["con_pdf_path"] = None
    st.session_state["con_pdf_error"] = None
    st.session_state["con_pdf_lock"] = False

def _background_worker(report_title, modules_payload, out_path):
    """Worker invoked in separate thread."""
    st.session_state["con_pdf_status"] = "running"
    ok, err = build_pdf_to_path(report_title, modules_payload, out_path)
    if ok:
        st.session_state["con_pdf_status"] = "ready"
        st.session_state["con_pdf_path"] = out_path
        st.session_state["con_pdf_error"] = None
    else:
        st.session_state["con_pdf_status"] = "error"
        st.session_state["con_pdf_error"] = err
    st.session_state["con_pdf_lock"] = False

# -------------------------
# Trigger UI
# -------------------------
out_file = os.path.join(TMP_PDF_DIR, f"People_Analytics_Leadership_Deck_{int(time.time())}.pdf")

col_gen, col_info = st.columns([3,1])
with col_gen:
    if st.button("🧾 Generate Consolidated Leadership Deck", use_container_width=True) and not st.session_state["con_pdf_lock"]:
        # prevent double clicks
        st.session_state["con_pdf_lock"] = True
        st.session_state["con_pdf_status"] = "queued"
        st.session_state["con_pdf_error"] = None
        # start background thread
        thread = threading.Thread(target=_background_worker, args=("People Analytics Leadership Deck", modules_payload, out_file), daemon=True)
        thread.start()
        st.info("⚙️ PDF generation queued in background — this may take some seconds. Stay on the page to see status.")

with col_info:
    status = st.session_state["con_pdf_status"]
    if status == "idle":
        st.write("Status: idle")
    elif status == "queued":
        st.write("Status: queued")
    elif status == "running":
        st.write("Status: running… (building PDF)")
    elif status == "ready":
        st.success("✅ PDF ready")
    elif status == "error":
        st.error("❌ PDF generation failed")

# If ready, show download button; if error, show trace
if st.session_state["con_pdf_status"] == "ready" and st.session_state["con_pdf_path"]:
    try:
        with open(st.session_state["con_pdf_path"], "rb") as f:
            pdf_bytes = f.read()
        st.download_button("⬇️ Download Leadership Deck (PDF)", pdf_bytes, file_name=os.path.basename(st.session_state["con_pdf_path"]), mime="application/pdf")
    except Exception as e:
        st.error(f"⚠️ Failed to load generated PDF: {e}")

if st.session_state["con_pdf_status"] == "error":
    st.text_area("Error details (traceback)", value=st.session_state.get("con_pdf_error", "No details"), height=300)