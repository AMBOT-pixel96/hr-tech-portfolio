# ============================================
# utils/pdf_explainer_builder.py — v5.2 | People Analytics Explainer (Boardroom Gradient Edition)
# ============================================
"""
Generates the People Analytics Executive Explainer PDF
with full gradient cover/thank-you pages, wrapped tables,
and confidentiality footer.
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
        "subtitle": ParagraphStyle("subtitle", parent=base["Heading2"],
                                   fontName=DEFAULT_FONT, fontSize=12,
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
# Zebra table (wrapped)
# ----------------------------
def make_zebra_table(data, col_widths):
    wrapped = []
    s = _get_styles()
    for r in data:
        wrapped.append([Paragraph(str(c), s["body"]) for c in r])

    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),INDIGO),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#D1D5DB")),
        ("FONTNAME",(0,0),(-1,-1),DEFAULT_FONT),
        ("WORDWRAP",(0,0),(-1,-1),"CJK"),
        ("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT_BG]),
    ]))
    return t

# ----------------------------
# Footer watermark
# ----------------------------
def _add_footer(c, doc):
    c.saveState()
    c.setFont(DEFAULT_FONT,8)
    c.setFillColor(GRAY_TEXT)
    w,_ = A4
    c.drawCentredString(w/2,12,"Prepared with ❤️ by People Analytics Project — Confidential")
    c.restoreState()

# ----------------------------
# Gradient background (cover / thank-you)
# ----------------------------
def draw_gradient_background(c):
    w,h = A4
    steps = 200
    for i in range(steps):
        ratio = i/steps
        r = NAVY.red + (INDIGO.red - NAVY.red) * ratio
        g = NAVY.green + (INDIGO.green - NAVY.green) * ratio
        b = NAVY.blue + (INDIGO.blue - NAVY.blue) * ratio
        c.setFillColorRGB(r,g,b)
        c.rect(0, (h/steps)*i, w, h/steps, stroke=0, fill=1)

# ----------------------------
# Minimal content (trimmed)
# ----------------------------
EXPLAINER_CONTENT = {
    "Attrition Analysis": {
        "blurb": "Understanding who’s leaving, how fast, and why.",
        "required": ["EmployeeID","Department","JobLevel","TenureMonths","AttritionFlag"],
        "sample": [["E301","Finance","L2","26","Yes"],["E302","Tech","L3","40","No"]],
        "metrics": [
            ("Attrition %","(Employees Left / Total) × 100","Workforce stability indicator."),
            ("Average Tenure","Σ(TenureMonths)/N","Average duration of stay."),
        ],
    },
    "Workforce Analysis": {
        "blurb":"Structural anatomy of your organization.",
        "required":["EmployeeID","JobLevel","Gender"],
        "sample":[["E001","L1","Male"],["E002","L2","Female"]],
        "metrics":[
            ("Total Headcount","COUNT(EmployeeID)","Total employees."),
            ("Female %","Count(Female)/Total×100","Gender balance."),
        ]
    },
}
# ----------------------------
# Main builder
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

    # ---------------------------------------------------
    # Cover (drawn via onFirstPage callback)
    # ---------------------------------------------------
    def cover(canvas, doc):
        draw_gradient_background(canvas)
        canvas.saveState()
        canvas.setFont(DEFAULT_FONT,24)
        canvas.setFillColor(colors.white)
        w,h = A4
        canvas.drawCentredString(w/2, h/2 + 30, "People Analytics Executive Explainer")
        canvas.setFont(DEFAULT_FONT,12)
        canvas.drawCentredString(w/2, h/2 - 10, f"Generated on {datetime.now().strftime('%d %b %Y')}")
        canvas.restoreState()

    # ---------------------------------------------------
    # TOC
    # ---------------------------------------------------
    story.append(Paragraph("📚 Table of Contents", s["heading"]))
    toc = [["#","Section","Description"],
           ["1","Module Overview","Purpose & Outputs"],
           ["2","Module Details","Required Fields, Samples & Metrics"],
           ["3","System Logics","Automation & Consolidation Flow"],
           ["4","Thank You","Confidentiality & Closure"]]
    story.append(make_zebra_table(toc,[10*mm,60*mm,110*mm]))
    story.append(PageBreak())

    # ---------------------------------------------------
    # Modules
    # ---------------------------------------------------
    for name,p in EXPLAINER_CONTENT.items():
        story.append(Paragraph(f"📘 {name}", s["heading"]))
        story.append(Paragraph(p["blurb"], s["body"]))
        story.append(Spacer(1,6))
        story.append(Paragraph("<b>Required Columns</b>", s["subhead"]))
        story.append(Paragraph(", ".join(p["required"]), s["body"]))
        story.append(Spacer(1,6))
        story.append(Paragraph("<b>Sample Rows</b>", s["subhead"]))
        story.append(make_zebra_table([p["required"]] + p["sample"], [40*mm]*len(p["required"])))
        story.append(Spacer(1,6))
        story.append(Paragraph("<b>Metrics & Formulas</b>", s["subhead"]))
        story.append(make_zebra_table([["Metric","Formula","Explanation"]]+p["metrics"], [45*mm,50*mm,70*mm]))
        story.append(PageBreak())

    # ---------------------------------------------------
    # Thank-you (drawn via onLaterPages callback)
    # ---------------------------------------------------
    def thank_you(canvas, doc):
        draw_gradient_background(canvas)
        canvas.saveState()
        canvas.setFont(DEFAULT_FONT,26)
        canvas.setFillColor(colors.white)
        w,h = A4
        canvas.drawCentredString(w/2, h/2, "Thank You")
        canvas.setFont(DEFAULT_FONT,9)
        canvas.drawCentredString(
            w/2, h/2 - 25,
            "This document and all data are confidential and intended for internal leadership review only."
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=cover, onLaterPages=lambda c,d: (thank_you(c,d), _add_footer(c,d)))
    pdf = buf.getvalue()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or "/tmp", exist_ok=True)
        with open(output_path,"wb") as f: f.write(pdf)
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