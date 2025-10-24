# modules/consolidated_module.py — v6.1 | Crash-Proof Consolidated Deck (File-based background builder)
import os, io, time, threading, json, traceback
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from utils_consolidated.chart_consolidated_saver import ensure_chart_saved
from utils_consolidated.uploader_consolidated_helper import upload_data

TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)
STATUS_FILE = os.path.join(TMP_DIR, "consolidated_status.json")

try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    FONT_NAME = "DejaVuSans"
except:
    FONT_NAME = "Helvetica"

PRIMARY_COLOR = colors.HexColor("#1E3A8A")
TABLE_HEADER_BG = colors.HexColor("#E5E7EB")
BODY_TEXT = colors.HexColor("#111827")

st.markdown("""
<div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
<h2 style="margin:0">📘 Consolidated HR Leadership Deck</h2>
<p style="margin:4px 0 0 0;">Background PDF generation (crash-proof mode)</p>
</div>
""", unsafe_allow_html=True)

# ------------------ Upload ------------------
st.markdown("### Upload All Module Datasets")
c1, c2, c3 = st.columns(3)
attr_df = upload_data("📉 Attrition Data", key="a")
comp_df = upload_data("💰 Compensation Data", key="b")
perf_df = upload_data("🏆 Performance Data", key="c")
c4, c5 = st.columns(2)
eng_df = upload_data("💬 Engagement Data", key="d")
work_df = upload_data("🏢 Workforce Data", key="e")

if not all(df is not None for df in [attr_df, comp_df, perf_df, eng_df, work_df]):
    st.info("📥 Upload all five datasets to proceed.")
    st.stop()

# ------------------ Helper ------------------
def _round_df(df, n=2):
    df2 = df.copy()
    for c in df2.select_dtypes(include=["float", "int"]).columns:
        df2[c] = df2[c].round(n)
    return df2

def _add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0]/2, 15, "Prepared with ❤️ by People Analytics Project — 2025")
    canvas.restoreState()

# ------------------ Build PDF ------------------
def build_pdf(out_path, modules_payload):
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        heading = ParagraphStyle("Heading", fontName=FONT_NAME, fontSize=14, textColor=PRIMARY_COLOR, spaceAfter=6)
        body = ParagraphStyle("Body", fontName=FONT_NAME, fontSize=10, textColor=BODY_TEXT)

        story = [Paragraph("<b>People Analytics Leadership Deck</b>", heading), Spacer(1,12)]
        for mod in modules_payload:
            story.append(Paragraph(f"<b>{mod['module_name']}</b>: {mod['module_desc']}", body))
            story.append(Spacer(1,6))
            for block in mod["data_blocks"]:
                story.append(Paragraph(block.get("title",""), heading))
                if block["df"] is not None:
                    df = _round_df(block["df"]).astype(str)
                    data = [list(df.columns)] + df.values.tolist()
                    table = Table(data, colWidths=[(A4[0]-40)/len(df.columns)]*len(df.columns))
                    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black),("BACKGROUND",(0,0),(-1,0),TABLE_HEADER_BG)]))
                    story.append(table)
                    story.append(Spacer(1,8))
                if block["fig"] is not None:
                    img = ensure_chart_saved(block["title"], block["fig"])
                    if img: story.append(RLImage(img, width=170*mm, height=95*mm))
            story.append(PageBreak())

        doc.build(story, onLaterPages=_add_footer)
        with open(out_path, "wb") as f:
            f.write(buf.getvalue())
        with open(STATUS_FILE, "w") as s:
            json.dump({"status": "ready", "path": out_path}, s)
    except Exception as e:
        with open(STATUS_FILE, "w") as s:
            json.dump({"status": "error", "error": str(e)}, s)

# ------------------ Prepare Modules ------------------
def build_payloads():
    mods = []
    if "AttritionFlag" in attr_df.columns:
        rate = (attr_df["AttritionFlag"].astype(str).str.lower().isin(["yes","y","1","true"]).mean())*100
        attr_blocks = [{"title":"Attrition Overview","desc":"Overall rate","df":pd.DataFrame({"Attrition%":[rate]}),"fig":None}]
    else: attr_blocks=[]
    mods.append({"module_name":"Attrition","module_desc":"Turnover trends","data_blocks":attr_blocks})
    mods.append({"module_name":"Workforce","module_desc":"Structure","data_blocks":[{"title":"Headcount","df":work_df.head(10),"fig":None}]})
    return mods

modules_payload = build_payloads()

# ------------------ Thread Starter ------------------
out_path = os.path.join(TMP_DIR, f"People_Analytics_Deck_{int(time.time())}.pdf")

def bg_worker():
    with open(STATUS_FILE, "w") as s:
        json.dump({"status": "running"}, s)
    build_pdf(out_path, modules_payload)

if st.button("🧾 Generate Consolidated Leadership Deck"):
    threading.Thread(target=bg_worker, daemon=True).start()
    st.info("⚙️ PDF generation started in background…")

# ------------------ Poller ------------------
if os.path.exists(STATUS_FILE):
    with open(STATUS_FILE) as f:
        status = json.load(f)
    if status.get("status") == "running":
        st.info("⏳ Building PDF… please wait.")
        st.experimental_rerun()
    elif status.get("status") == "ready":
        with open(status["path"], "rb") as f:
            st.success("✅ PDF ready!")
            st.download_button("⬇️ Download Leadership Deck", f, file_name=os.path.basename(status["path"]))
    elif status.get("status") == "error":
        st.error(f"⚠️ Error: {status.get('error')}")