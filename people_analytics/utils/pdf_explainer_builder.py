# ============================================
# utils/pdf_explainer_builder.py — v5.3 | People Analytics Explainer (Boardroom Gradient Edition)
# ============================================
"""
Generates the People Analytics Executive Explainer PDF
with clean layout, white-on-blue cover, wordwrapped tables,
and a single gradient Thank You page with footer.
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
# Font setup
# ----------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = "Helvetica"

# ----------------------------
# Colors
# ----------------------------
NAVY = colors.HexColor("#0F172A")
INDIGO = colors.HexColor("#1E3A8A")
LIGHT_BG = colors.HexColor("#F9FAFB")
GRAY_TEXT = colors.HexColor("#6B7280")

PAGE_LEFT_RIGHT_MARGIN = 18 * mm
PAGE_TOP_BOTTOM_MARGIN = 20 * mm

# ----------------------------
# Styles
# ----------------------------
def _get_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontName=DEFAULT_FONT, fontSize=24,
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
# Gradient background
# ----------------------------
def draw_gradient_background(c):
    w, h = A4
    steps = 100
    for i in range(steps):
        ratio = i / steps
        r = 15/255 + (30/255 - 15/255) * ratio
        g = 23/255 + (58/255 - 23/255) * ratio
        b = 42/255 + (138/255 - 42/255) * ratio
        c.setFillColorRGB(r, g, b)
        c.rect(0, i * (h / steps), w, (h / steps), stroke=0, fill=1)

# ----------------------------
# Footer watermark
# ----------------------------
def _add_footer(c, doc=None):
    c.saveState()
    c.setFont(DEFAULT_FONT, 8)
    c.setFillColor(GRAY_TEXT)
    w, _ = A4
    c.drawCentredString(w / 2, 12, "Prepared with ❤️ by People Analytics Project — Confidential")
    c.restoreState()

# ----------------------------
# Zebra Table
# ----------------------------
def make_zebra_table(data, col_widths):
    s = _get_styles()
    wrapped = [[Paragraph(str(c), s["body"]) for c in row] for row in data]
    t = Table(wrapped, colWidths=col_widths, repeatRows=1, splitByRow=True)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
        ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    return t

# ----------------------------
# Module content
# ----------------------------
EXPLAINER_CONTENT = {
    "Attrition Analysis": {
        "blurb": "Understanding who’s leaving, how fast, and why.",
        "required": ["EmployeeID", "Department", "JobLevel", "TenureMonths", "AttritionFlag"],
        "sample": [["E301", "Finance", "L2", "26", "Yes"], ["E302", "Tech", "L3", "40", "No"], ["E303", "HR", "L1", "15", "Yes"]],
        "metrics": [
            ("Attrition %", "(Employees Left / Total) × 100", "Percentage of employees who left during a specific period."),
            ("Average Tenure (months)", "Σ(TenureMonths) / N", "Average duration employees stay before leaving."),
            ("Attrition % by Department", "(DeptLeft / DeptTotal) × 100", "Compares exit rates across teams to spot turnover hot zones."),
            ("Attrition % by Job Level", "(LevelLeft / LevelTotal) × 100", "Reveals which hierarchy levels lose people fastest."),
            ("Attrition % by Tenure Cohort", "Grouped Tenure Cohort % Left", "Uncovers early-leaver patterns."),
            ("Exit Reasons (counts/share)", "Count(Reason)/TotalExits", "Breakdown of why employees left."),
        ],
    }
}

# ----------------------------
# Main Builder
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

    # ----- Cover -----
    def cover(c, doc):
        draw_gradient_background(c)
        c.saveState()
        c.setFont(DEFAULT_FONT, 26)
        c.setFillColor(colors.white)
        w, h = A4
        c.drawCentredString(w/2, h/2 + 30, "People Analytics Executive Explainer")
        c.setFont(DEFAULT_FONT, 12)
        c.drawCentredString(w/2, h/2 - 10, f"Generated on {datetime.now().strftime('%d %b %Y')}")
        c.restoreState()

    # ----- Table of Contents -----
    story.append(PageBreak())
    story.append(Paragraph("📚 Table of Contents", s["heading"]))
    toc = [["#", "Section", "Description"],
           ["1", "Module Overview", "Purpose & Outputs"],
           ["2", "Module Details", "Required Fields, Samples & Metrics"],
           ["3", "System Logics", "Automation & Consolidation Flow"],
           ["4", "Thank You", "Confidentiality & Closure"]]
    story.append(make_zebra_table(toc, [10*mm, 60*mm, 110*mm]))
    story.append(PageBreak())

    # ----- Modules -----
    for name, p in EXPLAINER_CONTENT.items():
        story.append(Paragraph(f"{name}", s["heading"]))
        story.append(Paragraph(p["blurb"], s["body"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Required Columns</b>", s["subhead"]))
        story.append(Paragraph(", ".join(p["required"]), s["body"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Sample Rows</b>", s["subhead"]))
        story.append(make_zebra_table([p["required"]] + p["sample"], [35*mm]*len(p["required"])))
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Metrics & Formulas</b>", s["subhead"]))
        story.append(make_zebra_table([["Metric", "Formula", "Explanation"]] + p["metrics"], [45*mm, 50*mm, 70*mm]))
        story.append(PageBreak())

    # ----- Build main doc -----
    doc.build(story, onFirstPage=cover, onLaterPages=_add_footer)

    # ----- Thank You Page -----
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    draw_gradient_background(c)
    c.setFont(DEFAULT_FONT, 28)
    c.setFillColor(colors.white)
    w, h = A4
    c.drawCentredString(w / 2, h / 2, "Thank You")
    _add_footer(c)
    c.showPage()
    c.save()
    packet.seek(0)
    pdf_thank = packet.getvalue()

    # Merge
    main_reader = PdfReader(io.BytesIO(buf.getvalue()))
    thank_reader = PdfReader(io.BytesIO(pdf_thank))
    writer = PdfWriter()
    for p in main_reader.pages: writer.add_page(p)
    for p in thank_reader.pages: writer.add_page(p)
    out = io.BytesIO(); writer.write(out)
    pdf = out.getvalue()

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