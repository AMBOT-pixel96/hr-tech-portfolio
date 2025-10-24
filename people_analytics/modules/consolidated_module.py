# modules/consolidated_module.py
# v1.0 synchronous, robust consolidated builder (no background threads)
import os
import io
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px

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

# local utils (ensure these paths exist in your repo)
from utils_consolidated.chart_consolidated_saver import ensure_chart_saved
from utils_consolidated.uploader_consolidated_helper import upload_data

# ----------------------------
# Logging / diagnostics
# ----------------------------
TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)
LOG_PATH = os.path.join(TMP_DIR, "consolidated_build.log")

def log(msg: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except Exception:
        # best-effort logging; don't crash the app
        pass

# ----------------------------
# Font setup for PDF
# ----------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    FONT_NAME = "DejaVuSans"
except Exception:
    FONT_NAME = "Helvetica"

PRIMARY_COLOR = colors.HexColor("#1E3A8A")
TABLE_HEADER_BG = colors.HexColor("#E5E7EB")
BODY_TEXT = colors.HexColor("#111827")

# ----------------------------
# Page header (UI)
# ----------------------------
st.markdown(
    """
<div style="padding:14px;border-radius:8px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
  <h2 style="margin:0">📘 Consolidated HR Leadership Deck</h2>
  <p style="margin:4px 0 0 0;">Upload module data and generate a boardroom-ready PDF.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### 🧩 Upload All Module Datasets")
st.caption("Supported formats: CSV, XLS, XLSX. Max recommended size per file: 200MB")

# ----------------------------
# Upload inputs
# ----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    attr_df = upload_data("📉 Attrition Data", key="con_attr")
with c2:
    comp_df = upload_data("💰 Compensation Data", key="con_comp")
with c3:
    perf_df = upload_data("🏆 Performance Data", key="con_perf")

c4, c5 = st.columns(2)
with c4:
    eng_df = upload_data("💬 Engagement Data", key="con_eng")
with c5:
    work_df = upload_data("🏢 Workforce Data", key="con_work")

if not all(df is not None for df in [attr_df, comp_df, perf_df, eng_df, work_df]):
    st.info("📥 Please upload all five datasets to proceed.")
    st.stop()

# ----------------------------
# Small helpers
# ----------------------------
def _round_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float", "int"]).columns:
        df2[c] = df2[c].round(decimals)
    return df2

def _add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0] / 2, 15, "Prepared with ❤️ by People Analytics Project — 2025")
    canvas.restoreState()

# ----------------------------
# Build data blocks (mirrors previous design)
# ----------------------------
log("Preparing module payloads")
try:
    # Attrition
    attr_blocks = []
    if isinstance(attr_df, pd.DataFrame) and "AttritionFlag" in attr_df.columns:
        attr_rate = (attr_df["AttritionFlag"].astype(str).str.lower().isin(["yes","y","1","true"]).mean()) * 100
        avg_tenure = attr_df["TenureMonths"].mean() if "TenureMonths" in attr_df else None
        dept = (
            attr_df.groupby("Department", observed=True)["AttritionFlag"]
            .apply(lambda x: (x.astype(str).str.lower().isin(["yes","y","1","true"])).mean() * 100)
            .reset_index(name="AttritionRate")
        )
        fig_attr = px.bar(dept, x="Department", y="AttritionRate", color="Department", title="Attrition % by Department")
        attr_blocks = [
            {"title": "Attrition Overview", "desc": "Overall attrition metrics",
             "df": pd.DataFrame({"Attrition %": [round(attr_rate, 2)], "Avg Tenure (mo)": [round(avg_tenure, 2) if avg_tenure else "N/A"]}),
             "fig": None, "insights": [f"Attrition rate: {attr_rate:.1f}%", f"Avg tenure: {avg_tenure:.1f} mo" if avg_tenure else "N/A"]},
            {"title": "Departmental Attrition", "desc": "Attrition % by Department",
             "df": _round_df(dept), "fig": fig_attr, "insights": []},
        ]
    # Compensation
    comp_blocks = []
    if isinstance(comp_df, pd.DataFrame) and "CTC" in comp_df.columns:
        comp_df["CTC"] = pd.to_numeric(comp_df["CTC"], errors="coerce")
        comp_df["Bonus"] = pd.to_numeric(comp_df.get("Bonus", 0), errors="coerce")
        comp_df["BonusPct"] = (comp_df["Bonus"] / comp_df["CTC"].replace(0, pd.NA)) * 100
        ctc = comp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index(name="AvgCTC")
        bonus = comp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index(name="AvgBonusPct")
        fig_ctc = px.bar(ctc, x="JobLevel", y="AvgCTC", color="JobLevel", title="Avg CTC by Job Level")
        fig_bonus = px.bar(bonus, x="JobLevel", y="AvgBonusPct", color="JobLevel", title="Bonus % by Job Level")
        comp_blocks = [
            {"title": "CTC by Job Level", "desc": "Average internal pay per level", "df": _round_df(ctc), "fig": fig_ctc},
            {"title": "Bonus by Job Level", "desc": "Average bonus % per level", "df": _round_df(bonus), "fig": fig_bonus},
        ]
    # Performance
    perf_blocks = []
    if isinstance(perf_df, pd.DataFrame) and "PerformanceRating" in perf_df.columns:
        perf_df["PerformanceRating"] = pd.to_numeric(perf_df["PerformanceRating"], errors="coerce")
        job_perf = perf_df.groupby("JobLevel", observed=True)["PerformanceRating"].mean().reset_index(name="AvgRating")
        fig_perf = px.bar(job_perf, x="JobLevel", y="AvgRating", color="JobLevel", title="Avg Performance Rating by Job Level")
        perf_blocks = [{"title": "Performance Summary", "desc": "Average rating per job level", "df": _round_df(job_perf), "fig": fig_perf}]
    # Engagement
    eng_blocks = []
    qcols = [c for c in eng_df.columns if c.upper().startswith("Q")]
    if qcols:
        eng_df[qcols] = eng_df[qcols].apply(pd.to_numeric, errors="coerce")
        eng_df["EngagementIndex"] = eng_df[qcols].mean(axis=1)
        dept_eng = eng_df.groupby("Department", observed=True)["EngagementIndex"].mean().reset_index(name="MeanIndex")
        fig_eng = px.bar(dept_eng, x="Department", y="MeanIndex", color="Department", title="Engagement Index by Department")
        eng_blocks = [
            {"title": "Engagement Overview", "desc": "Overall engagement index", "df": pd.DataFrame({"Average Index": [eng_df['EngagementIndex'].mean().round(2)]}), "fig": None},
            {"title": "Departmental Engagement", "desc": "Avg engagement by department", "df": _round_df(dept_eng), "fig": fig_eng},
        ]
    # Workforce
    work_blocks = []
    if isinstance(work_df, pd.DataFrame) and "JobLevel" in work_df.columns:
        headcount = work_df.groupby("JobLevel", observed=True).size().reset_index(name="Headcount")
        fig_hc = px.bar(headcount, x="JobLevel", y="Headcount", color="JobLevel", title="Headcount by Job Level")
        gender_split = work_df["Gender"].value_counts(normalize=True).mul(100).reset_index()
        gender_split.columns = ["Gender", "Percent"]
        fig_gender = px.pie(gender_split, names="Gender", values="Percent", title="Gender Composition")
        work_blocks = [
            {"title": "Headcount Structure", "desc": "Employee count by level", "df": headcount, "fig": fig_hc},
            {"title": "Gender Composition", "desc": "Gender % across workforce", "df": gender_split, "fig": fig_gender},
        ]

    modules_payload = [
        {"module_name": "Attrition", "module_desc": "Turnover & tenure trends", "data_blocks": attr_blocks},
        {"module_name": "Compensation", "module_desc": "Pay & benchmarking", "data_blocks": comp_blocks},
        {"module_name": "Performance", "module_desc": "Performance distribution & KPIs", "data_blocks": perf_blocks},
        {"module_name": "Engagement", "module_desc": "Survey sentiment & participation", "data_blocks": eng_blocks},
        {"module_name": "Workforce", "module_desc": "Structure & diversity insights", "data_blocks": work_blocks},
    ]
except Exception as e:
    log(f"Error preparing payloads: {e}\n{traceback.format_exc()}")
    st.error("⚠️ Failed preparing the module payloads. See logs: /tmp/consolidated_build.log")
    st.stop()

# ----------------------------
# PDF render function (synchronous)
# ----------------------------
def render_consolidated_pdf(report_title: str, modules_payload: list, filename_prefix: str):
    """Synchronous build. Shows progress via spinner and surfaces errors explicitly."""
    if not modules_payload:
        st.warning("⚠️ No module data available.")
        return

    out_file = os.path.join(TMP_DIR, f"{filename_prefix}_{int(datetime.now().timestamp())}.pdf")
    try:
        with st.spinner("🧾 Building PDF — this can take 10-60s depending on charts..."):
            # Build
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle("Title", fontName=FONT_NAME, fontSize=20, alignment=1, textColor=colors.HexColor("#0F172A"))
            heading = ParagraphStyle("Heading", fontName=FONT_NAME, fontSize=14, textColor=PRIMARY_COLOR, spaceAfter=6)
            body = ParagraphStyle("Body", fontName=FONT_NAME, fontSize=10, textColor=BODY_TEXT)

            story = []
            # Cover
            story.append(Spacer(1, 30))
            story.append(Paragraph(f"<b>{report_title}</b>", title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}", body))
            story.append(PageBreak())

            # Per-module pages
            for mod in modules_payload:
                story.append(Paragraph(f"{mod.get('module_name','')}", heading))
                story.append(Paragraph(mod.get("module_desc",""), body))
                story.append(Spacer(1, 8))
                for block in mod.get("data_blocks", []):
                    story.append(Paragraph(block.get("title", ""), heading))
                    if block.get("df") is not None and not block.get("df").empty:
                        df = block["df"].round(2).astype(str)
                        tbl_data = [list(df.columns)] + df.values.tolist()
                        table = Table(tbl_data, colWidths=[(A4[0] - 40) / max(1, len(df.columns))] * len(df.columns))
                        table.setStyle(TableStyle([
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                            ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 8))
                    if block.get("fig") is not None:
                        img_path = ensure_chart_saved(block.get("title", "chart"), block.get("fig"))
                        if img_path:
                            story.append(RLImage(img_path, width=170*mm, height=95*mm))
                            story.append(Spacer(1, 8))
                        else:
                            story.append(Paragraph("⚠️ Chart could not be rendered.", body))
                story.append(PageBreak())

            doc.build(story, onLaterPages=_add_footer)
            pdf_bytes = buf.getvalue()

            # Write out file
            with open(out_file, "wb") as f:
                f.write(pdf_bytes)
            buf.close()

        # Success
        st.success("✅ Consolidated Leadership Deck built!")
        with open(out_file, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(out_file), mime="application/pdf")
        log(f"PDF built successfully at {out_file}")

    except Exception as e:
        log(f"PDF build error: {e}\n{traceback.format_exc()}")
        st.error("⚠️ PDF build failed. See logs at /tmp/consolidated_build.log")
        st.write(f"Error detail (first 500 chars): {str(e)[:500]}")

# ----------------------------
# Page UI: Generate button
# ----------------------------
st.markdown("---")
st.header("📄 Generate Consolidated Executive Report")
st.caption("Combine all uploaded modules into a single PDF.")

if st.button("🧾 Generate Consolidated Leadership Deck", use_container_width=True):
    # call synchronously — errors will be shown (no blank screen)
    render_consolidated_pdf("People Analytics Leadership Deck", modules_payload, "People_Analytics_Deck")