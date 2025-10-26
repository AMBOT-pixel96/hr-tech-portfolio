# ============================================
# utils_consolidated/pdf_merger.py — v6.0 | Dual-Pipeline Stable (Final Form)
# ============================================
import os
from PyPDF2 import PdfMerger
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from datetime import datetime
from io import BytesIO
from utils_consolidated.pdf_helper_consolidated import extract_summary_table

TMP_DIR = "/tmp/consolidated_pdfs"

def merge_consolidated_pdfs(output_path):
    os.makedirs(TMP_DIR, exist_ok=True)
    pdfs = [os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(output_path)
    merger.close()

def generate_final_consolidated_pdf(output_path="Final_Leadership_Deck.pdf"):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=14, textColor=colors.HexColor("#1E3A8A"), spaceAfter=8)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14)

    story = []

    # --- Consolidated TOC ---
    story.append(Paragraph("📘 Consolidated Table of Contents", heading))
    toc_data = [["Module", "Metrics Overview"]]
    summaries = extract_summary_table(TMP_DIR)
    for s in summaries:
        toc_data.append([s["Module"], s["Metrics"]])
    toc_table = Table(toc_data, colWidths=[40*mm, 130*mm])
    toc_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E5E7EB")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.black),
        ("WORDWRAP", (1,1), (-1,-1), "CJK"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")])
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # --- Add all modules ---
    merger = PdfMerger()
    for pdf in [os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]:
        merger.append(pdf)
    temp_merge = os.path.join(TMP_DIR, "_temp_merged.pdf")
    merger.write(temp_merge)
    merger.close()

    # --- Consolidated Executive Summary ---
    story.append(Paragraph("🧠 Consolidated Executive Summary", heading))
    summary_data = [["Module", "Key Insights"]]
    for s in summaries:
        summary_data.append([s["Module"], s["Insights"]])
    summary_table = Table(summary_data, colWidths=[40*mm, 130*mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E5E7EB")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.black),
        ("WORDWRAP", (1,1), (-1,-1), "CJK"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")])
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    # --- Thank You Page ---
    story.append(Spacer(1, 120))
    story.append(Paragraph("<para align=center><font size=22 color='#1E3A8A'><b>Thank You</b></font></para>", body))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<para align=center><font size=12>People Analytics Leadership Deck</font></para>", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<para align=center><font size=10 color='#6B7280'>© 2025 People Analytics Project — Confidential</font></para>", body))

    doc.build(story)
    final_data = buf.getvalue()
    with open(output_path, "wb") as f:
        f.write(final_data)

    return output_path