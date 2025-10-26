# ============================================
# utils_consolidated/pdf_merger.py — v7.2 | Case-Insensitive Stable Build
# ============================================
import os
import io
from datetime import datetime
from PyPDF2 import PdfMerger
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from utils_consolidated.pdf_helper_consolidated import extract_summary_table

TMP_DIR = "/mount/src/hr-tech-portfolio/people_analytics/tmp" if os.path.exists(
    "/mount/src/hr-tech-portfolio/people_analytics/tmp"
) else "/tmp"

def merge_consolidated_pdfs(output_path):
    """
    Merges all module PDFs into one consolidated executive deck.
    Adds Consolidated TOC + Executive Summary + Thank You page.
    """
    expected_modules = ["Attrition", "Compensation", "Engagement", "Performance", "Workforce"]

    pdf_files = {
        m.lower(): os.path.join(TMP_DIR, f)
        for f in os.listdir(TMP_DIR)
        for m in expected_modules
        if f.lower().startswith(m.lower()) and f.lower().endswith(".pdf")
    }

    # Debug log
    print(f"🔍 Found PDFs: {list(pdf_files.keys())}")

    missing = [m for m in expected_modules if m.lower() not in pdf_files]
    if missing:
        raise FileNotFoundError(f"Missing module PDFs: {', '.join(missing)}")

    merger = PdfMerger()

    # ===============================
    # 🧩 1️⃣ Add Consolidated TOC Page
    # ===============================
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#1E3A8A"), spaceAfter=6)
    normal = ParagraphStyle("normal", parent=styles["Normal"], fontSize=10)

    story = []
    story.append(Paragraph("📘 Consolidated Table of Contents", heading))
    story.append(Spacer(1, 6))

    summaries = extract_summary_table(TMP_DIR)
    toc_data = [["Module", "Metrics Overview"]]
    for s in summaries:
        toc_data.append([s["Module"], s["Metrics"]])

    toc_table = Table(toc_data, colWidths=[40 * mm, 130 * mm], repeatRows=1)
    toc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    merger.append(buf)

    # ===============================
    # 🧩 2️⃣ Add Module PDFs (Cleaned)
    # ===============================
    for module in expected_modules:
        pdf_path = pdf_files.get(module.lower())
        if pdf_path and os.path.exists(pdf_path):
            print(f"📄 Adding section: {module}")
            merger.append(pdf_path)

    # ===============================
    # 🧩 3️⃣ Add Consolidated Executive Summary
    # ===============================
    summary_buf = io.BytesIO()
    doc = SimpleDocTemplate(summary_buf, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=20 * mm, bottomMargin=20 * mm)
    story = []

    story.append(Paragraph("🧠 Consolidated Executive Summary", heading))
    story.append(Spacer(1, 6))

    summary_data = [["Module", "Key Insights"]]
    for s in summaries:
        summary_data.append([s["Module"], s["Insights"]])

    summary_table = Table(summary_data, colWidths=[40 * mm, 130 * mm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    story.append(summary_table)
    story.append(PageBreak())
    doc.build(story)
    summary_buf.seek(0)
    merger.append(summary_buf)

    # ===============================
    # 🧩 4️⃣ Add Thank You Page
    # ===============================
    thanks_buf = io.BytesIO()
    doc = SimpleDocTemplate(thanks_buf, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=20 * mm, bottomMargin=20 * mm)
    story = []

    story.append(Spacer(1, 200))
    story.append(Paragraph(
        "<para align=center><font size=24 color='#1E3A8A'><b>Thank You</b></font></para>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<para align=center><font size=12 color='#374151'>Prepared with ❤️ by Amlan Mishra<br/>People Analytics Leadership Deck</font></para>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 50))
    story.append(Paragraph(
        "<para align=center><font size=9 color='#9CA3AF'>© 2025 People Analytics Project — Confidential</font></para>",
        styles["Normal"]
    ))

    doc.build(story)
    thanks_buf.seek(0)
    merger.append(thanks_buf)

    # ===============================
    # 💾 Export Final PDF
    # ===============================
    with open(output_path, "wb") as f:
        merger.write(f)
    merger.close()
    print(f"✅ Consolidated Leadership Deck created: {output_path}")
    return True