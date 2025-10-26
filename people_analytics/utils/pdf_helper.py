# ============================================
# utils/pdf_helper.py — v5.1 | Executive Polish Edition (Production)
# ============================================
import os
import io
import streamlit as st
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from utils.chart_saver import ensure_chart_saved

try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
except:
    pass
DEFAULT_FONT = "DejaVuSans"

def render_pdf_download_button(report_title, module_name, data_blocks, file_prefix):
    if not data_blocks:
        st.warning("⚠️ No data blocks available.")
        return

    if st.button(f"🧾 Generate {module_name} Executive PDF", use_container_width=True):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)

        styles = getSampleStyleSheet()
        body = ParagraphStyle("body", parent=styles["Normal"], fontName=DEFAULT_FONT, fontSize=10, leading=14)
        heading = ParagraphStyle("heading", parent=styles["Heading2"], fontName=DEFAULT_FONT, fontSize=13, textColor=colors.HexColor("#1E3A8A"), spaceAfter=6)
        story = []

        # Cover Page
        story.append(Spacer(1, 100))
        story.append(Paragraph(f"<para align=center><font size=22><b>{report_title}</b></font></para>", body))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"<para align=center><font size=13 color='#374151'>{module_name} Module</font></para>", body))
        story.append(PageBreak())

        # TOC (without page numbers)
        toc_data = [["#", "Section", "Description"]]
        for i, block in enumerate(data_blocks, 1):
            toc_data.append([i, block.get("title", ""), block.get("desc", "")])
        table = Table(toc_data, colWidths=[20*mm, 60*mm, 95*mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E5E7EB")),
            ("GRID", (0,0), (-1,-1), 0.25, colors.black)
        ]))
        story.append(Paragraph("<b>Table of Contents</b>", heading))
        story.append(table)
        story.append(PageBreak())

        # Section content
        summary_data = [["Section", "Key Insights"]]
        for i, block in enumerate(data_blocks, 1):
            title, desc, df, fig, insights = (
                block.get("title"), block.get("desc"), block.get("df"), block.get("fig"), block.get("insights")
            )
            if not insights:
                continue  # skip empty insight blocks

            story.append(Paragraph(f"{i}. {title}", heading))
            story.append(Paragraph(desc, body))
            if fig is not None:
                img = ensure_chart_saved(title, fig)
                if img and os.path.exists(img):
                    story.append(RLImage(img, width=175*mm, height=105*mm))
            story.append(Spacer(1, 6))
            story.append(Paragraph(" • ".join(insights), body))