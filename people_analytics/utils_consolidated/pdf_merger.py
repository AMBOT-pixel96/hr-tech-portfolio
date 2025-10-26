# ============================================
# utils_consolidated/pdf_merger.py — v7.5 | Executive Boardroom Edition (Font + Aesthetic Fix)
# ============================================
import os
import io
import shutil
from datetime import datetime
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from utils_consolidated.pdf_helper_consolidated import extract_summary_table

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
# 🎨 Helper: Create gradient divider/cover with shadow text
# -------------------------------------------------------
def _make_gradient_page(title: str, subtitle: str = "", color1="#0F172A", color2="#1E3A8A", add_watermark=True):
    """Generates a single stylish gradient divider/cover page with subtle text shadow and watermark."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Force UTF-8 & DejaVuSans for all draw operations
    c.setFont(FONT_NAME, 28)
    title = title.encode("utf-8", "ignore").decode("utf-8")
    subtitle = subtitle.encode("utf-8", "ignore").decode("utf-8")

    # Gradient background simulation
    for i in range(100):
        blend = i / 100.0
        r1, g1, b1 = colors.HexColor(color1).rgb()
        r2, g2, b2 = colors.HexColor(color2).rgb()
        r = r1 + (r2 - r1) * blend
        g = g1 + (g2 - g1) * blend
        b = b1 + (b2 - b1) * blend
        c.setFillColorRGB(r, g, b)
        c.rect(0, i * (height / 100), width, height / 100, stroke=0, fill=1)

    # Shadowed Title
    c.setFont(FONT_NAME, 28)
    # Shadow layer
    c.setFillColor(colors.HexColor("#000000"))
    c.drawCentredString(width / 2 + 1.5, height / 2 + 10 * mm - 1.5, title)
    # Main text
    c.setFillColor(colors.white)
    c.drawCentredString(width / 2, height / 2 + 10 * mm, title)

    # Shadowed Subtitle
    if subtitle:
        c.setFont(FONT_NAME, 14)
        c.setFillColor(colors.HexColor("#000000"))
        c.drawCentredString(width / 2 + 1, height / 2 - 10 * mm - 1, subtitle)
        c.setFillColor(colors.HexColor("#FACC15"))
        c.drawCentredString(width / 2, height / 2 - 10 * mm, subtitle)

    # Footer watermark
    if add_watermark:
        c.setFont(FONT_NAME, 9)
        c.setFillColor(colors.HexColor("#E5E7EB"))
        c.drawCentredString(width / 2, 15, "People Analytics 2025")

    c.showPage()
    c.save()
    return buf.getvalue()

# -------------------------------------------------------
# 📦 Merge PDFs into one executive deck
# -------------------------------------------------------
def merge_consolidated_pdfs(output_filename: str = "People_Analytics_Leadership_Deck.pdf"):
    """Merges all module PDFs from TMP_DIR into one, with dividers, TOC, exec summary, and a styled thank-you page."""
    st.markdown("### 🧩 Consolidation Summary")

    pdf_files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        st.info("⚙️ No PDFs found in consolidated queue. Add modules first.")
        return False

    pdf_files.sort()
    output_path = os.path.join(TMP_DIR, output_filename)
    writer = PdfWriter()

    # -------------------------------------------------------
    # 🧠 Cover Page
    # -------------------------------------------------------
    st.write("📘 Adding cover page...")
    cover_pdf = PdfReader(io.BytesIO(_make_gradient_page(
        "People Analytics Leadership Deck",
        f"Generated {datetime.now().strftime('%b’%y')}",
        add_watermark=True
    )))
    writer.append(cover_pdf)

    # -------------------------------------------------------
    # 📖 Consolidated Table of Contents
    # -------------------------------------------------------
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontName=FONT_NAME, fontSize=14, textColor=colors.HexColor("#1E3A8A"))
    small = ParagraphStyle("Small", parent=styles["Normal"], fontName=FONT_NAME, fontSize=9, leading=12, textColor=colors.black)

    toc_data = [["Module", "Metrics", "Key Insights"]]
    summaries = extract_summary_table(TMP_DIR)
    for s in summaries:
        toc_data.append([
            Paragraph(f"<b>{s['Module']}</b>", small),
            Paragraph(s["Metrics"], small),
            Paragraph(s["Insights"] or "—", small)
        ])

    table = Table(toc_data, colWidths=[40*mm, 60*mm, 70*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))

    story = [Paragraph("<b>Consolidated Table of Contents</b>", heading), Spacer(1, 8), table]
    doc.build(story)
    toc_pdf = PdfReader(io.BytesIO(buf.getvalue()))
    writer.append(toc_pdf)

    # -------------------------------------------------------
    # 🧩 Add each module PDF with stylish divider
    # -------------------------------------------------------
    for f in pdf_files:
        section_name = os.path.splitext(f)[0].replace("_", " ")
        st.write(f"📄 Adding section: {section_name}")
        divider_pdf = PdfReader(io.BytesIO(_make_gradient_page(section_name.title(), "Module Summary", add_watermark=True)))
        writer.append(divider_pdf)

        try:
            reader = PdfReader(os.path.join(TMP_DIR, f))
            for page in reader.pages[1:]:  # Skip module cover
                writer.add_page(page)
        except Exception as e:
            st.error(f"⚠️ Could not merge {f}: {e}")

    # -------------------------------------------------------
    # 🧾 Consolidated Executive Summary
    # -------------------------------------------------------
    buf2 = io.BytesIO()
    doc2 = SimpleDocTemplate(buf2, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm)
    exec_data = [["Module", "Key Insights"]]
    for s in summaries:
        exec_data.append([
            Paragraph(f"<b>{s['Module']}</b>", small),
            Paragraph(s["Insights"] or "—", small)
        ])
    exec_table = Table(exec_data, colWidths=[45*mm, 125*mm])
    exec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))
    story2 = [Paragraph("<b>Consolidated Executive Summary</b>", heading), Spacer(1, 8), exec_table]
    doc2.build(story2)
    exec_pdf = PdfReader(io.BytesIO(buf2.getvalue()))
    writer.append(exec_pdf)

    # -------------------------------------------------------
    # 💌 Thank You Page (clean alignment, font + watermark)
    # -------------------------------------------------------
    thank_buf = io.BytesIO()
    c = canvas.Canvas(thank_buf, pagesize=A4)
    width, height = A4
    c.setFillColor(colors.HexColor("#0F172A"))
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFont(FONT_NAME, 36)
    c.setFillColor(colors.white)
    c.drawCentredString(width / 2, height / 2 + 20 * mm, "THANK YOU")
    c.setFont(FONT_NAME, 14)
    c.setFillColor(colors.HexColor("#FACC15"))
    c.drawCentredString(width / 2, height / 2 - 10 * mm, "For reviewing the People Analytics Leadership Deck")
    c.setFont(FONT_NAME, 9)
    c.setFillColor(colors.HexColor("#E5E7EB"))
    c.drawCentredString(width / 2, 20, "© 2025 People Analytics Project — Confidential")
    c.showPage()
    c.save()
    thank_pdf = PdfReader(io.BytesIO(thank_buf.getvalue()))
    writer.append(thank_pdf)

    # -------------------------------------------------------
    # 💾 Save + Download
    # -------------------------------------------------------
    with open(output_path, "wb") as out:
        writer.write(out)

    st.success(f"✅ Consolidated Leadership Deck created successfully: {output_filename}")
    with open(output_path, "rb") as f:
        st.download_button(
            "⬇️ Download Final Consolidated Deck",
            f,
            file_name=output_filename,
            mime="application/pdf"
        )

    return True