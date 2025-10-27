# ============================================
# utils/pdf_explainer_builder.py — v5.8 | People Analytics Explainer (Boardroom Gold Edition)
# ============================================
"""
Final production version:
✅ Clean separate cover (no TOC overlap)
✅ White-on-indigo table headers
✅ Non-stretched sample rows
✅ Confidentiality disclaimer box on gradient
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
# Gradient background
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
# Footer
# ----------------------------
def _add_footer(c, doc):
    c.saveState()
    c.setFont(DEFAULT_FONT, 8)
    c.setFillColor(GRAY_TEXT)
    w, _ = A4
    c.drawCentredString(w / 2, 12, "Prepared with ❤️ by People Analytics Project — Confidential")
    c.restoreState()

# ----------------------------
# Zebra table with white header text
# ----------------------------
def make_zebra_table(data, col_widths):
    s = _get_styles()
    wrapped = [[Paragraph(str(c), s["body"]) for c in r] for r in data]

    # ✅ Force white text for header row
    for j in range(len(wrapped[0])):
        wrapped[0][j] = Paragraph(f"<font color='white'><b>{data[0][j]}</b></font>", s["body"])

    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
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
# Explainer content dictionary
# ----------------------------
EXPLAINER_CONTENT = {
    "Attrition Analysis": {
        "blurb": "Understanding who’s leaving, how fast, and why.",
        "required": ["EmployeeID", "Department", "JobLevel", "TenureMonths", "AttritionFlag"],
        "sample": [["E301", "Finance", "L2", "26", "Yes"],
                   ["E302", "Tech", "L3", "40", "No"],
                   ["E303", "HR", "L1", "15", "Yes"]],
        "metrics": [
            ("Attrition %", "(Employees Left / Total) × 100", "Percentage of employees who left during a period."),
            ("Average Tenure (months)", "Σ(TenureMonths) / N", "Average stay duration before leaving."),
            ("Attrition % by Department", "(DeptLeft / DeptTotal) × 100", "Compares turnover across teams."),
            ("Attrition % by Job Level", "(LevelLeft / LevelTotal) × 100", "Reveals hierarchy churn."),
            ("Attrition % by Tenure Cohort", "Grouped Tenure Cohort % Left", "Shows early-leaver patterns."),
            ("Exit Reasons (counts/share)", "Count(Reason)/TotalExits", "Categorical reason analysis."),
        ],
    },
    "Compensation Analysis": {
        "blurb": "Understanding pay fairness, competitiveness, and motivation levers.",
        "required": ["EmployeeID", "Department", "CTC", "Bonus", "JobLevel", "Gender"],
        "sample": [["E201", "Tech", "1500000", "150000", "L3", "Male"],
                   ["E202", "Finance", "900000", "75000", "L2", "Female"],
                   ["E203", "HR", "700000", "50000", "L1", "Female"]],
        "metrics": [
            ("Average CTC", "Σ(CTC)/N", "Mean total annual cost per employee."),
            ("Average Bonus %", "Mean(Bonus/CTC×100)", "Average variable pay ratio."),
            ("Bonus % by Job Level", "Avg(Bonus/CTC×100) per Level", "Spread across hierarchy."),
            ("Avg CTC by Job Level", "Avg(CTC) by Level", "Pay progression across levels."),
            ("Avg CTC by Gender", "Avg(CTC) by Gender", "Checks pay equity."),
            ("Internal vs Market", "Compare AvgCTC vs MarketMedian", "Internal vs external benchmark."),
            ("Market Gap % by Job Level", "(AvgCTC–MarketMedian)/MarketMedian×100", "Gap to market median."),
        ],
    },
    "Engagement Analysis": {
        "blurb": "How emotionally and mentally invested employees feel at work.",
        "required": ["Department", "Gender", "Q1", "Q2", "Q3", "Q4"],
        "sample": [["Sales", "Male", "4", "3", "5", "4"],
                   ["HR", "Female", "5", "4", "4", "5"],
                   ["Tech", "Male", "3", "4", "3", "4"]],
        "metrics": [
            ("Engagement Index", "Mean(Q1..Qn)", "Composite score of survey responses."),
            ("Avg Engagement Index", "Avg(EngagementIndex)", "Mean engagement score."),
            ("Highly Engaged %", "Count(Index>3.6)/Total×100", "Top-tier engagement ratio."),
            ("Low Engaged %", "Count(Index≤2.9)/Total×100", "Disengaged proportion."),
            ("Engagement Index by Department", "Avg(Index) per Department", "Team-level culture score."),
            ("Engagement Categories", "Bucket Index into ranges", "Distribution of sentiment."),
            ("Engagement by Gender", "Avg(Index) by Gender", "Gender-level engagement gap."),
        ],
    },
    "Performance Analysis": {
        "blurb": "How people perform and whether rewards are fair.",
        "required": ["EmployeeID", "Department", "CTC", "PerformanceRating"],
        "sample": [["E101", "Finance", "950000", "4"],
                   ["E102", "Tech", "1200000", "5"],
                   ["E103", "HR", "800000", "3"]],
        "metrics": [
            ("Average Rating", "Σ(PerformanceRating)/N", "Mean rating across employees."),
            ("Rating SD", "StdDev(PerformanceRating)", "Variance in performance scores."),
            ("Avg Rating by Dept", "Avg(Rating) per Department", "Team-wise performance."),
            ("Avg Rating by Level", "Avg(Rating) per Level", "Hierarchical differentiation."),
            ("Top Performers %", "Count(Rating≥4)/Total×100", "High-performer ratio."),
            ("Low Performers %", "Count(Rating≤2)/Total×100", "Low-performer share."),
            ("Performance KDE", "Density of Ratings", "Shape of performance curve."),
            ("Performance vs Pay", "Correlation(CTC, Rating)", "Pay-for-performance linkage."),
            ("Gender Performance", "Avg(Rating) by Gender", "Bias detection metric."),
        ],
    },
    "Workforce Analysis": {
        "blurb": "The structural anatomy of your organization.",
        "required": ["EmployeeID", "JobLevel", "Gender"],
        "sample": [["E001", "L1", "Male"],
                   ["E002", "L2", "Female"],
                   ["E003", "L3", "Male"]],
        "metrics": [
            ("Total Headcount", "COUNT(EmployeeID)", "Total workforce size."),
            ("Headcount by Level", "Count(EmployeeID) per Level", "Hierarchical structure."),
            ("Female %", "Count(Female)/Total×100", "Gender ratio."),
            ("Number of Job Levels", "Distinct(JobLevel)", "Total hierarchy layers."),
            ("Manager Span", "Avg(DirectReports per Manager)", "Leadership bandwidth."),
            ("Top Manager Spans", "Top N managers by direct reports", "Load distribution."),
            ("Skill Inventory", "Tokenize & Count(Skills)", "Top emerging skills."),
        ],
    },
}

# ----------------------------
# Build PDF
# ----------------------------
def build_explainer_pdf(output_path=None) -> bytes:
    s = _get_styles()

    # --- Create cover separately
    cover_buf = io.BytesIO()
    c = canvas.Canvas(cover_buf, pagesize=A4)
    draw_gradient_background(c)
    w, h = A4
    c.setFont(DEFAULT_FONT, 26)
    c.setFillColor(colors.white)
    c.drawCentredString(w / 2, h / 2 + 30, "People Analytics Executive Explainer")
    c.setFont(DEFAULT_FONT, 12)
    c.drawCentredString(w / 2, h / 2 - 10, f"Generated on {datetime.now().strftime('%d %b %Y')}")
    c.showPage()
    c.save()

    # --- TOC + modules ---
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        rightMargin=PAGE_LEFT_RIGHT_MARGIN,
        leftMargin=PAGE_LEFT_RIGHT_MARGIN,
        topMargin=PAGE_TOP_BOTTOM_MARGIN,
        bottomMargin=PAGE_TOP_BOTTOM_MARGIN,
    )
    story = []
    story.append(Paragraph("Table of Contents", s["heading"]))
    toc = [["#", "Section", "Description"],
           ["1", "Module Overview", "Purpose & Outputs"],
           ["2", "Module Details", "Required Fields, Samples & Metrics"],
           ["3", "System Logics", "Automation & Consolidation Flow"],
           ["4", "Thank You", "Confidentiality & Closure"]]
    story.append(make_zebra_table(toc, [10 * mm, 60 * mm, 110 * mm]))
    story.append(PageBreak())

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
        story.append(make_zebra_table([["Metric", "Formula", "Explanation"]] + p["metrics"],
                                      [45 * mm, 50 * mm, 70 * mm]))
        story.append(PageBreak())

    doc.build(story, onLaterPages=_add_footer)

    # --- Thank You page ---
    packet = io.BytesIO()
    tcan = canvas.Canvas(packet, pagesize=A4)
    draw_gradient_background(tcan)
    tcan.setFont(DEFAULT_FONT, 28)
    tcan.setFillColor(colors.white)
    tcan.drawCentredString(A4[0] / 2, A4[1] / 2 + 10, "Thank You")
    tcan.setStrokeColor(YELLOW_BORDER)
    tcan.rect(50, 110, A4[0] - 100, 85, stroke=1, fill=0)
    tcan.setFont(DEFAULT_FONT, 9)
    tcan.setFillColor(YELLOW_BORDER)
    y = 180
    for line in [
        "⚠️ Confidentiality Note",
        "• System for internal use only.",
        "• No personally identifiable data is stored or transmitted.",
        "• Reports are confidential leadership artifacts."
    ]:
        tcan.drawCentredString(A4[0] / 2, y, line)
        y -= 15
    tcan.setFont(DEFAULT_FONT, 8)
    tcan.setFillColor(GRAY_TEXT)
    tcan.drawCentredString(A4[0] / 2, 25, "Prepared with ❤️ by People Analytics Project — Confidential")
    tcan.showPage()
    tcan.save()

    # --- Merge cover + body + thank you ---
    writer = PdfWriter()
    for src in [cover_buf, buf, packet]:
        src.seek(0)
        reader = PdfReader(src)
        for p in reader.pages:
            writer.add_page(p)

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