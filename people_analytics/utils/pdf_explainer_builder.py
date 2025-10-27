# ============================================
# utils/pdf_explainer_builder.py — v5.0 | People Analytics Explainer (Boardroom Arc)
# ============================================
"""
Generates the People Analytics Executive Explainer PDF.

Features:
- Cover page (matches consolidated deck look)
- Manual TOC (zebra table, no page numbers)
- Module overview + required columns + sample rows
- Metric table per module (Metric, Formula, Description)
- Upload guidance, tips & limitations
- System logic & interactive features (chatbot, sequencing, mailbox)
- Confidentiality clause
- Thank-you page (matches consolidated deck look)
- Footer watermark on every page
"""

import os
import io
from datetime import datetime
import streamlit as st
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ----------------------------
# Font registration (best-effort)
# ----------------------------
try:
    pdfmetrics.registerFont(
        TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = "Helvetica"

# ----------------------------
# Colors / layout constants
# ----------------------------
PRIMARY = colors.HexColor("#1E3A8A")   # Indigo
NAVY = colors.HexColor("#0F172A")      # Deep navy
ACCENT = colors.HexColor("#2563EB")
LIGHT_BG = colors.HexColor("#F9FAFB")
TOC_HEADER_BG = colors.HexColor("#E5E7EB")

PAGE_LEFT_RIGHT_MARGIN = 18 * mm
PAGE_TOP_BOTTOM_MARGIN = 20 * mm
# ----------------------------
# Content definition
# ----------------------------
EXPLAINER_CONTENT = {
    "Attrition Analysis": {
        "blurb": "Understanding who’s leaving, how fast, and why.",
        "required": ["EmployeeID", "Department", "JobLevel", "TenureMonths", "AttritionFlag"],
        "sample": [["E301","Finance","L2","26","Yes"],
                   ["E302","Tech","L3","40","No"],
                   ["E303","HR","L1","15","Yes"]],
        "metrics": [
            ("Attrition %", "(Employees Left / Total) × 100",
             "Percentage of employees who left during a period."),
            ("Average Tenure (months)", "Σ(TenureMonths) / N",
             "Average duration employees stay before leaving."),
            ("Attrition % by Department", "(DeptLeft / DeptTotal) × 100",
             "Compares exit rates across departments."),
            ("Attrition % by Job Level", "(LevelLeft / LevelTotal) × 100",
             "Shows which hierarchy levels lose people fastest."),
            ("Exit Reasons", "Count(Reason)/TotalExits",
             "Categorizes reasons employees left.")
        ],
    },
    "Compensation Analysis": {
        "blurb": "Understanding pay fairness, competitiveness, and motivation levers.",
        "required": ["EmployeeID","Department","CTC","Bonus","JobLevel","Gender"],
        "sample": [["E201","Tech","1500000","150000","L3","Male"],
                   ["E202","Finance","900000","75000","L2","Female"],
                   ["E203","HR","700000","50000","L1","Female"]],
        "metrics": [
            ("Average CTC","Σ(CTC)/N","Mean total annual cost per employee."),
            ("Average Bonus %","Mean(Bonus/CTC×100)","Average variable pay ratio."),
            ("Bonus % by Job Level","Avg(Bonus/CTC×100) per Level",
             "Compares incentive spread across hierarchy."),
            ("Avg CTC by Gender","Avg(CTC) by Gender",
             "Checks gender pay parity."),
            ("Market Gap %","(AvgCTC–MarketMedian)/MarketMedian×100",
             "Benchmarks vs market median.")
        ],
    },
    "Engagement Analysis": {
        "blurb": "How emotionally and mentally invested employees feel at work.",
        "required": ["Department","Gender","Q1","Q2","Q3","Q4"],
        "sample": [["Sales","Male","4","3","5","4"],
                   ["HR","Female","5","4","4","5"],
                   ["Tech","Male","3","4","3","4"]],
        "metrics": [
            ("Engagement Index","Mean(Q1..Qn)",
             "Composite score of survey responses."),
            ("Highly Engaged %","Count(Index>3.6)/Total×100",
             "Proportion scoring in top tier."),
            ("Low Engaged %","Count(Index≤2.9)/Total×100",
             "Share of disengaged employees."),
            ("Engagement by Department","Avg(Index) per Dept",
             "Cultural differences across teams.")
        ],
    },
    "Performance Analysis": {
        "blurb": "How people perform and whether rewards are fair.",
        "required": ["EmployeeID","Department","CTC","PerformanceRating"],
        "sample": [["E101","Finance","950000","4"],
                   ["E102","Tech","1200000","5"],
                   ["E103","HR","800000","3"]],
        "metrics": [
            ("Avg Rating","Σ(Rating)/N","Average performance rating."),
            ("Rating StdDev","StdDev(Rating)","Spread of ratings."),
            ("Top Performers %","Count(Rating≥4)/Total×100",
             "Share of high performers."),
            ("Performance vs Pay","Corr(CTC, Rating)",
             "Pay-performance alignment.")
        ],
    },
    "Workforce Analysis": {
        "blurb": "The structural anatomy of your organization.",
        "required": ["EmployeeID","JobLevel","Gender"],
        "sample": [["E001","L1","Male"],
                   ["E002","L2","Female"],
                   ["E003","L3","Male"]],
        "metrics": [
            ("Total Headcount","COUNT(EmployeeID)","Total employees."),
            ("Female %","Count(Female)/Total×100","Gender ratio."),
            ("Job Levels","Distinct(JobLevel)","Hierarchy count."),
            ("Manager Span","Avg(DirectReports per Manager)",
             "Span of control metric.")
        ],
    },
}

# ----------------------------
# Styles & helpers
# ----------------------------
def _get_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontName=DEFAULT_FONT, fontSize=20,
                                alignment=1, textColor=PRIMARY),
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                             fontName=DEFAULT_FONT, fontSize=14,
                             textColor=PRIMARY, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                             fontName=DEFAULT_FONT, fontSize=11,
                             textColor=ACCENT, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontName=DEFAULT_FONT, fontSize=9,
                               leading=12),
        "footer": ParagraphStyle("footer", fontName=DEFAULT_FONT,
                                 fontSize=8, alignment=1,
                                 textColor=colors.HexColor("#6B7280")),
    }

def make_zebra_table(data, col_widths):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),PRIMARY),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.25,colors.black),
        ("FONTNAME",(0,0),(-1,-1),DEFAULT_FONT),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT_BG]),
    ]))
    return t
# ----------------------------
# Footer watermark
# ----------------------------
def _add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(DEFAULT_FONT, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0]/2, 12,
        "Prepared with ❤️ by People Analytics Project — Confidential")
    canvas.restoreState()

# ----------------------------
# PDF builder
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
    s = _get_styles(); story = []

    # Cover Page
    story.append(Spacer(1,100))
    story.append(Paragraph(
        "<b>People Analytics Leadership System — Executive Explainer</b>",
        s["title"]))
    story.append(Spacer(1,12))
    story.append(Paragraph(
        "Data formats • Metric formulas • Logic explanations • Confidential guide",
        s["h2"]))
    story.append(Spacer(1,200))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d %b %Y')}", s["footer"]))
    story.append(PageBreak())

    # TOC
    toc=[["#","Section","Description"],
         ["1","Module Overview","Purpose & Outputs"],
         ["2","Module Details","Required Fields & Metrics"],
         ["3","System Logics","Automation and Workflow"],
         ["4","Confidentiality","Usage Guidelines"],
         ["5","Thank You","Closing Note"]]
    story.append(make_zebra_table(toc,[10*mm,55*mm,110*mm]))
    story.append(PageBreak())

    # Module Overview
    story.append(Paragraph("📑 Module Overview",s["h1"]))
    overview=[["Module","Purpose","Outputs"],
              ["Workforce","Headcount & Gender Insights","Workforce Report"],
              ["Performance","Ratings & Pay Correlation","Performance Report"],
              ["Engagement","Survey & Sentiment Index","Engagement Report"],
              ["Compensation","Pay & Market Benchmarking","Comp Report"],
              ["Attrition","Exits & Tenure Patterns","Attrition Report"]]
    story.append(make_zebra_table(overview,[40*mm,70*mm,65*mm]))
    story.append(PageBreak())

    # Per-module sections
    for name,p in EXPLAINER_CONTENT.items():
        story.append(Paragraph(f"📘 {name}",s["h1"]))
        story.append(Paragraph(p["blurb"],s["body"]))
        story.append(Spacer(1,6))
        story.append(Paragraph("<b>Required Columns</b>",s["h2"]))
        story.append(Paragraph(", ".join(p["required"]),s["body"]))
        story.append(Spacer(1,6))
        story.append(Paragraph("<b>Sample Rows</b>",s["h2"]))
        story.append(make_zebra_table([p["required"]]+p["sample"],[35*mm]*len(p["required"])))
        story.append(Spacer(1,6))
        story.append(Paragraph("<b>Metrics & Formulas</b>",s["h2"]))
        m=[["Metric","Formula","Explanation"]]+p["metrics"]
        story.append(make_zebra_table(m,[45*mm,50*mm,70*mm]))
        story.append(PageBreak())

    # Confidentiality and Thank You
    story.append(Paragraph("⚠️ Confidentiality Note",s["h1"]))
    for n in [
        "System for internal use only.",
        "No personally identifiable data is stored or transmitted.",
        "Reports are confidential leadership artifacts."
    ]:
        story.append(Paragraph("• "+n,s["body"]))
    story.append(PageBreak())
    story.append(Spacer(1,200))
    story.append(Paragraph(
        "<para align=center><font size=22 color='#1E3A8A'><b>Thank You</b></font></para>",
        s["body"]))
    story.append(Spacer(1,10))
    story.append(Paragraph(
        "<para align=center><font size=11 color='#374151'>For reviewing the People Analytics Leadership System.</font></para>",
        s["body"]))
    story.append(Spacer(1,50))
    story.append(Paragraph("© 2025 People Analytics Project — Confidential",s["footer"]))

    doc.build(story,onLaterPages=_add_footer)
    pdf=buf.getvalue()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or "/tmp",exist_ok=True)
        with open(output_path,"wb") as f:f.write(pdf)
    return pdf
def show_explainer_ui():
    st.header("📘 People Analytics Explainer PDF")
    st.caption("Comprehensive explainer for all modules, metrics and logic.")
    if st.button("📄 Generate Explainer PDF"):
        try:
            pdf = build_explainer_pdf()
            st.download_button(
                "⬇️ Download Explainer PDF",
                pdf,
                "People_Analytics_Explainer.pdf",
                "application/pdf",
            )
        except Exception as e:
            st.error(f"⚠️ PDF generation failed: {e}")

# Local test
if __name__ == "__main__":
    b = build_explainer_pdf("/tmp/People_Analytics_Explainer_v5.pdf")
    print("Wrote /tmp/People_Analytics_Explainer_v5.pdf", len(b), "bytes")

