# ============================================
# utils/pdf_explainer_builder.py — v1.2 | People Analytics Leadership System Explainer (No Spill Jutsu)
# ============================================
"""
📘 Purpose:
Generates the *official explainer document* for the People Analytics Leadership System.
This PDF serves as a one-stop reference guide for:
- All modules (Workforce, Performance, Engagement, Compensation, Attrition)
- Upload templates and data formats
- Metric definitions and insights
- System logic, export pipelines, and chatbot + sequencing integration

✅ Spill-proof tables (no overflowing text)
✅ Consistent enterprise design with zebra styling
✅ Single entry-point callable from app.py
"""

import os
import io
from datetime import datetime
import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -----------------------------------------------------------
# 🧩 Font Setup (Unicode + ₹ + symbols)
# -----------------------------------------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
except:
    pass
DEFAULT_FONT = "DejaVuSans"


# -----------------------------------------------------------
# 🦓 Zebra Table Utility (Spill-Proof Edition)
# -----------------------------------------------------------
def make_zebra_table(data, col_widths):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    return table


# -----------------------------------------------------------
# 🧠 Main Explainer PDF Builder
# -----------------------------------------------------------
def generate_explainer_pdf():
    """Generates the official People Analytics Leadership System Explainer PDF."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", fontName=DEFAULT_FONT, fontSize=14, textColor=colors.HexColor("#1E3A8A"), spaceAfter=6)
    subhead = ParagraphStyle("Subhead", fontName=DEFAULT_FONT, fontSize=11, textColor=colors.HexColor("#0F172A"), spaceAfter=4)
    body = ParagraphStyle("Body", fontName=DEFAULT_FONT, fontSize=9, leading=12, textColor=colors.black)
    footer = ParagraphStyle("Footer", fontName=DEFAULT_FONT, fontSize=8, textColor=colors.HexColor("#6B7280"), alignment=1)

    story = []

    # ---------------------------------------------------
    # 🧠 COVER PAGE
    # ---------------------------------------------------
    story.append(Spacer(1, 100))
    story.append(Paragraph("<b>People Analytics Leadership System — Executive Explainer</b>", heading))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Your one-stop guide to building, merging, and mastering HR intelligence.", subhead))
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d-%b-%Y')}", body))
    story.append(Spacer(1, 200))
    story.append(Paragraph("© 2025 People Analytics Project — Confidential", footer))
    story.append(PageBreak())

    # ---------------------------------------------------
    # 📑 MODULE OVERVIEW
    # ---------------------------------------------------
    story.append(Paragraph("📑 Module Overview", heading))
    data = [
        ["Module", "Core Purpose", "Key Outputs"],
        ["Workforce", "Headcount, Gender Ratio, Spans, Skills", "Workforce Executive Report"],
        ["Performance", "Ratings, Pay Correlation", "Performance Executive Report"],
        ["Engagement", "Survey Metrics, Index, Sentiment", "Engagement Executive Report"],
        ["Compensation", "Pay Distribution, Market Benchmarks", "Compensation Executive Report"],
        ["Attrition", "Exit Reasons, Tenure Insights", "Attrition Executive Report"],
    ]
    story.append(make_zebra_table(data, [35 * mm, 80 * mm, 60 * mm]))
    story.append(PageBreak())

    # ---------------------------------------------------
    # 📂 MODULE DETAILS
    # ---------------------------------------------------
    module_specs = {
        "Workforce": {
            "Required Columns": ["EmployeeID", "JobLevel", "Gender"],
            "Optional Columns": ["ManagerID", "Skills"],
            "Metrics": ["Headcount", "Gender %", "Span of Control"],
        },
        "Performance": {
            "Required Columns": ["EmployeeID", "Department", "JobLevel", "CTC", "PerformanceRating"],
            "Optional Columns": ["Gender"],
            "Metrics": ["Avg Rating", "Top Performers %", "CTC vs Rating"],
        },
        "Engagement": {
            "Required Columns": ["Department", "Gender", "Q* (Survey Responses)"],
            "Optional Columns": [],
            "Metrics": ["Engagement Index", "Highly Engaged %", "Low Engaged %"],
        },
        "Compensation": {
            "Required Columns": ["EmployeeID", "Department", "Gender", "CTC", "Bonus", "JobLevel"],
            "Optional Columns": ["MarketMedianCTC"],
            "Metrics": ["Avg CTC", "Avg Bonus %", "Internal vs Market Comparison"],
        },
        "Attrition": {
            "Required Columns": ["EmployeeID", "Department", "JobLevel", "TenureMonths", "AttritionFlag"],
            "Optional Columns": ["ExitReason"],
            "Metrics": ["Attrition %", "Avg Tenure", "Top Exit Reasons"],
        }
    }

    for mod, details in module_specs.items():
        story.append(Paragraph(f"📘 {mod} Module", heading))
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Required Columns</b>", subhead))
        story.append(Paragraph(", ".join(details["Required Columns"]), body))
        story.append(Paragraph("<b>Optional Columns</b>", subhead))
        story.append(Paragraph(", ".join(details["Optional Columns"]) or "None", body))
        story.append(Paragraph("<b>Metrics Captured</b>", subhead))
        story.append(Paragraph(", ".join(details["Metrics"]), body))
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>File Formats</b>: CSV / XLSX (Recommended: XLSX for mobile users)", body))
        story.append(PageBreak())

    # ---------------------------------------------------
    # 🧠 SYSTEM LOGICS
    # ---------------------------------------------------
    story.append(Paragraph("🧠 System Logics", heading))
    logic_data = [
        ["Logic Type", "Description"],
        ["Executive PDF Generation", "Builds per-module boardroom-ready PDFs with branded styling."],
        ["Add to Consolidated", "Copies each module’s PDF + metadata JSON to /tmp/consolidated_pdfs."],
        ["Merge Consolidated Deck", "Combines all added PDFs into one unified HR Leadership Deck."],
        ["Consolidated Chatbot", "Lets users query their consolidated insights interactively."],
        ["Job Sequencing Engine", "Visualizes promotion and role progression using Sankey charts."],
    ]
    story.append(make_zebra_table(logic_data, [50 * mm, 120 * mm]))
    story.append(PageBreak())

    # ---------------------------------------------------
    # 🤖 FEATURE EXPLAINER
    # ---------------------------------------------------
    story.append(Paragraph("🤖 Interactive Features", heading))
    story.append(Paragraph("💬 Chatbot Assistant — Ask natural questions about metrics or insights.", body))
    story.append(Paragraph("📈 Job Sequencing — Visualize career progression by level and department.", body))
    story.append(Paragraph("📤 Send to Mailbox — Simulated e-mail confirmation workflow.", body))
    story.append(PageBreak())

    # ---------------------------------------------------
    # ⚠️ CONFIDENTIALITY
    # ---------------------------------------------------
    story.append(Paragraph("⚠️ Confidentiality & Usage", heading))
    story.append(Paragraph("""
        All materials in this dashboard and report suite are strictly confidential and intended for internal HR use only.<br/>
        No personal or identifiable employee information is stored or transmitted. 
        Generated reports are anonymized and stored temporarily within the Streamlit app session.
    """, body))
    story.append(PageBreak())

    # ---------------------------------------------------
    # 🎨 THANK YOU PAGE
    # ---------------------------------------------------
    story.append(Spacer(1, 200))
    story.append(Paragraph("<para align=center><font size=22 color='#1E3A8A'><b>Thank You</b></font></para>", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<para align=center><font size=11 color='#374151'>For reviewing the People Analytics Leadership System.</font></para>", body))
    story.append(Spacer(1, 150))
    story.append(Paragraph("© 2025 People Analytics Project — Confidential", footer))

    # ---------------------------------------------------
    # 💾 EXPORT
    # ---------------------------------------------------
    try:
        doc.build(story)
        pdf_bytes = buf.getvalue()
        st.success("✅ Explainer PDF generated successfully.")
        st.download_button(
            "⬇️ Download System Explainer PDF",
            pdf_bytes,
            file_name="People_Analytics_Explainer.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"⚠️ Failed to generate explainer PDF: {e}")
    finally:
        buf.close()