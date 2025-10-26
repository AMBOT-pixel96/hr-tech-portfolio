# ============================================
# utils_consolidated/pdf_merger.py — v3.0 | Executive Final Production Build
# ============================================
import os
import io
import json
import shutil
from datetime import datetime
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -------------------------------------------------------
# 🗂️ Directory setup
# -------------------------------------------------------
TMP_DIR = "/tmp/consolidated_pdfs"
os.makedirs(TMP_DIR, exist_ok=True)

# -------------------------------------------------------
# 🧠 Fonts
# -------------------------------------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    FONT_NAME = "DejaVuSans"
except Exception:
    FONT_NAME = "Helvetica"

# -------------------------------------------------------
# 📄 Helper: generate single-page PDF (cover or divider)
# -------------------------------------------------------
def _make_single_page_pdf(title: str, subtitle: str = "", color: str = "#1E3A8A") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    c.setFillColor(colors.HexColor(color))
    c.rect(0, 0, width, height, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont(FONT_NAME, 24)
    c.drawCentredString(width / 2, height / 2 + 10, title)
    if subtitle:
        c.setFont(FONT_NAME, 14)
        c.drawCentredString(width / 2, height / 2 - 20, subtitle)
    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#E5E7EB"))
    c.drawCentredString(width / 2, 15, "© 2025 People Analytics Project — Confidential")
    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 🧾 Merge PDFs into one executive deck
# -------------------------------------------------------
def merge_consolidated_pdfs(output_path: str = os.path.join(TMP_DIR, "People_Analytics_Leadership_Deck.pdf")) -> bool:
    pdf_files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        st.warning("⚙️ No PDFs found in consolidated queue. Add modules first.")
        return False

    pdf_files.sort()
    writer = PdfWriter()

    # Cover Page
    cover_pdf = PdfReader(io.BytesIO(_make_single_page_pdf(
        "People Analytics Leadership Deck",
        f"Generated {datetime.now().strftime('%b’%y')}",
        "#0F172A"
    )))
    writer.append(cover_pdf)

    # Add consolidated TOC
    toc_data = [["Module", "Metrics Overview"]]
    for pdf in pdf_files:
        module_name = pdf.replace(".pdf", "")
        json_path = os.path.join(TMP_DIR, f"{module_name}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            toc_data.append([module_name, meta.get("metrics_short", "—")])
        else:
            toc_data.append([module_name, "—"])

    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []
    heading = ParagraphStyle("Heading", fontName=FONT_NAME, fontSize=14, textColor=colors.HexColor("#0F172A"), spaceAfter=8)
    story.append(Paragraph("📘 Consolidated Table of Contents", heading))
    table = Table(toc_data, colWidths=[120, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 9)
    ]))
    story.append(table)
    doc.build(story)
    writer.append(PdfReader(io.BytesIO(buf.getvalue())))

    # Append each module
    for f in pdf_files:
        module_name = os.path.splitext(f)[0]
        divider_pdf = PdfReader(io.BytesIO(_make_single_page_pdf(module_name.title(), "Module Summary", "#1E3A8A")))
        writer.append(divider_pdf)
        reader = PdfReader(os.path.join(TMP_DIR, f))
        for i, page in enumerate(reader.pages):
            if i == 0:  # skip module cover page
                continue
            writer.add_page(page)

    # Consolidated Executive Summary
    summary_pdf = _generate_consolidated_summary(pdf_files)
    if summary_pdf:
        writer.append(PdfReader(io.BytesIO(summary_pdf)))

    # Thank You Page
    thank_you_pdf = PdfReader(io.BytesIO(_make_single_page_pdf("Thank You 💼", "Prepared with ❤️ by Amlan Mishra", "#0F172A")))
    writer.append(thank_you_pdf)

    # Write final file
    with open(output_path, "wb") as out:
        writer.write(out)
    return True

# -------------------------------------------------------
# 🧠 Consolidated Summary Generator
# -------------------------------------------------------
def _generate_consolidated_summary(pdf_files):
    summary_data = [["Module", "Key Insights"]]
    for pdf in pdf_files:
        module_name = pdf.replace(".pdf", "")
        json_path = os.path.join(TMP_DIR, f"{module_name}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            summary_data.append([module_name, meta.get("insights", "No summary provided.")])
        else:
            summary_data.append([module_name, "No summary provided."])

    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []
    heading = ParagraphStyle("Heading", fontName=FONT_NAME, fontSize=14, textColor=colors.HexColor("#0F172A"), spaceAfter=8)
    story.append(Paragraph("🧠 Consolidated Executive Summary", heading))
    table = Table(summary_data, colWidths=[120, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 9)
    ]))
    story.append(table)
    doc.build(story)
    return buf.getvalue()