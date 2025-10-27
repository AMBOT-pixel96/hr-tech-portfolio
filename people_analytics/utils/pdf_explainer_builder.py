# ============================================
# utils/pdf_explainer_builder.py — v5.6 | People Analytics Explainer (Boardroom Ultra Fixed Edition)
# ============================================
"""
Generates the People Analytics Executive Explainer PDF
with gradient cover, clean TOC, wrapped tables,
white header text, and confidentiality disclaimer box.
"""

import os, io
from datetime import datetime
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from PyPDF2 import PdfReader, PdfWriter


# ----------------------------
# Font registration (with fallback)
# ----------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = "Helvetica"

# ----------------------------
# Color palette
# ----------------------------
NAVY = colors.HexColor("#0F172A")
INDIGO = colors.HexColor("#1E3A8A")
LIGHT_BG = colors.HexColor("#F9FAFB")
GRAY_TEXT = colors.HexColor("#6B7280")
YELLOW_BORDER = colors.HexColor("#FACC15")

PAGE_LEFT_RIGHT_MARGIN = 18 * mm
PAGE_TOP_BOTTOM_MARGIN = 20 * mm


# ----------------------------
# Styles
# ----------------------------
def _get_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontName=DEFAULT_FONT, fontSize=22,
                                alignment=1, textColor=colors.white),
        "heading": ParagraphStyle("heading", parent=base["Heading1"],
                                  fontName=DEFAULT_FONT, fontSize=14,
                                  textColor=INDIGO, spaceAfter=6),
        "subhead": ParagraphStyle("subhead", parent=base["Heading2"],
                                  fontName=DEFAULT_FONT, fontSize=11,
                                  textColor=colors.HexColor("#1E40AF"), spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontName=DEFAULT_FONT, fontSize=9,
                               leading=12, textColor=colors.black),
        "footer": ParagraphStyle("footer", parent=base["Normal"],
                                 fontName=DEFAULT_FONT, fontSize=8,
                                 alignment=1, textColor=GRAY_TEXT)
    }


# ----------------------------
# Zebra Table (wrapped, white header text)
# ----------------------------
def make_zebra_table(data, col_widths):
    s = _get_styles()
    wrapped = [[Paragraph(str(c), s["body"]) for c in r] for r in data]

    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    return t


# ----------------------------
# Footer watermark
# ----------------------------
def _add_footer(c, doc):
    c.saveState()
    c.setFont(DEFAULT_FONT, 8)
    c.setFillColor(GRAY_TEXT)
    w, _ = A4
    c.drawCentredString(w / 2, 12, "Prepared with ❤️ by People Analytics Project — Confidential")
    c.restoreState()


# ----------------------------
# Gradient Background (cover / thank-you)
# ----------------------------
def draw_gradient_background(c):
    w, h = A4
    steps = 200
    for i in range(steps):
        ratio = i / steps
        r = NAVY.red + (INDIGO.red - NAVY.red) * ratio
        g = NAVY.green + (INDIGO.green - NAVY.green) * ratio
        b = NAVY.blue + (INDIGO.blue - NAVY.blue) * ratio
        c.setFillColorRGB(r, g, b)
        c.rect(0, (h / steps) * i, w, h / steps, stroke=0, fill=1)


# ----------------------------
# Explainer Content (unchanged)
# ----------------------------
EXPLAINER_CONTENT = {  # ... keep your full content block as-is ...
    # [same data as before, unchanged for brevity]
}


# ----------------------------
# Main PDF Builder
# ----------------------------
def build_explainer_pdf(output_path=None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=PAGE_LEFT_RIGHT_MARGIN,
        leftMargin=PAGE_LEFT_RIGHT_MARGIN,
        topMargin=PAGE_TOP_BOTTOM_MARGIN,
        bottomMargin=PAGE_TOP_BOTTOM_MARGIN,
    )
    s = _get_styles()
    story = []

    # --- Cover page ---
    def cover(canvas, doc):
        draw_gradient_background(canvas)
        w, h = A4
        canvas.saveState()
        canvas.setFont(DEFAULT_FONT, 26)
        canvas.setFillColor(colors.white)
        canvas.drawCentredString(w / 2, h / 2 + 30, "People Analytics Executive Explainer")
        canvas.setFont(DEFAULT_FONT, 12)
        canvas.drawCentredString(w / 2, h / 2 - 10, f"Generated on {datetime.now().strftime('%d %b %Y')}")
        canvas.restoreState()

    # --- Table of Contents ---
    story.append(PageBreak())
    story.append(Paragraph("Table of Contents", s["heading"]))
    toc = [["#", "Section", "Description"],
           ["1", "Module Overview", "Purpose & Outputs"],
           ["2", "Module Details", "Required Fields, Samples & Metrics"],
           ["3", "System Logics", "Automation & Consolidation Flow"],
           ["4", "Thank You", "Confidentiality & Closure"]]
    story.append(make_zebra_table(toc, [10 * mm, 60 * mm, 110 * mm]))
    story.append(PageBreak())

    # --- Module Sections ---
    for name, p in EXPLAINER_CONTENT.items():
        story.append(Paragraph(name, s["heading"]))
        story.append(Paragraph(p["blurb"], s["body"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Required Columns", s["subhead"]))
        story.append(Paragraph(", ".join(p["required"]), s["body"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Sample Rows", s["subhead"]))
        col_count = len(p["required"])
        col_width = (180 * mm) / col_count
        story.append(make_zebra_table([p["required"]] + p["sample"], [col_width] * col_count))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Metrics & Formulas", s["subhead"]))
        story.append(make_zebra_table([["Metric", "Formula", "Explanation"]] + p["metrics"], [45 * mm, 50 * mm, 70 * mm]))
        story.append(PageBreak())

    # --- Build main body ---
    doc.build(story, onFirstPage=cover, onLaterPages=_add_footer)

    # --- Thank You Page ---
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=A4)
    draw_gradient_background(can)
    w, h = A4
    can.setFont(DEFAULT_FONT, 28)
    can.setFillColor(colors.white)
    can.drawCentredString(w / 2, h / 2 + 10, "Thank You")

    # Confidentiality box
    can.setStrokeColor(YELLOW_BORDER)
    can.rect(50, 110, w - 100, 85, stroke=1, fill=0)
    can.setFont(DEFAULT_FONT, 9)
    can.setFillColor(YELLOW_BORDER)
    y = 180
    for line in [
        "Confidentiality Note",
        "• System for internal use only.",
        "• No personally identifiable data is stored or transmitted.",
        "• Reports are confidential leadership artifacts."
    ]:
        can.drawCentredString(w / 2, y, line)
        y -= 15

    # Footer
    can.setFont(DEFAULT_FONT, 8)
    can.setFillColor(GRAY_TEXT)
    can.drawCentredString(w / 2, 25, "Prepared with ❤️ by People Analytics Project — Confidential")

    can.showPage()
    can.save()

    # Merge PDFs
    main_reader = PdfReader(io.BytesIO(buf.getvalue()))
    thank_reader = PdfReader(io.BytesIO(packet.getvalue()))
    writer = PdfWriter()
    for page in main_reader.pages:
        writer.add_page(page)
    for page in thank_reader.pages:
        writer.add_page(page)

    output = io.BytesIO()
    writer.write(output)
    pdf = output.getvalue()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or "/tmp", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf)

    return pdf


# ----------------------------
# Streamlit UI
# ----------------------------
def show_explainer_ui():
    st.header("📘 People Analytics Executive Explainer")
    st.caption("Generate a gradient-themed, boardroom-ready PDF explaining every module and metric.")
    if st.button("📄 Generate Explainer PDF"):
        try:
            pdf = build_explainer_pdf()
            st.success("✅ Explainer PDF generated successfully.")
            st.download_button(
                "⬇️ Download Explainer PDF",
                pdf,
                file_name=f"People_Analytics_Explainer_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"⚠️ PDF generation failed: {e}")