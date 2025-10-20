# ============================================
# utils/pdf_helper.py — v2 Executive Edition
# ============================================

import os
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register DejaVuSans (emoji + ₹ support)
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
except:
    pass

styles = getSampleStyleSheet()
base_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="DejaVuSans",
    fontSize=10,
    leading=14,
    textColor=colors.black,
)

title_style = ParagraphStyle(
    "Title",
    parent=styles["Title"],
    fontName="DejaVuSans",
    fontSize=22,
    leading=28,
    alignment=1,
    textColor=colors.HexColor("#1E3A8A"),
)

subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontName="DejaVuSans",
    fontSize=14,
    alignment=1,
    textColor=colors.HexColor("#374151"),
)

section_style = ParagraphStyle(
    "Section",
    parent=styles["Heading2"],
    fontName="DejaVuSans",
    fontSize=16,
    textColor=colors.HexColor("#1E3A8A"),
    spaceAfter=8,
)

summary_style = ParagraphStyle(
    "Summary",
    parent=styles["Normal"],
    fontName="DejaVuSans",
    fontSize=10,
    textColor=colors.black,
)

# ============================================
# Helper: Create Zebra Table
# ============================================
def zebra_table(data, col_widths=None):
    """Creates a zebra-styled table."""
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),  # Header row
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9CA3AF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table

# ============================================
# Section 1: Cover Page
# ============================================
def cover_page(title: str, author: str, module_name: str):
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    return [
        Spacer(1, 120),
        Paragraph(title, title_style),
        Spacer(1, 10),
        Paragraph(f"<b>{module_name}</b>", subtitle_style),
        Spacer(1, 180),
        Paragraph(f"Prepared with ❤️ by <b>{author}</b>", base_style),
        Paragraph(f"Generated on {now}", base_style),
        PageBreak(),
    ]

# ============================================
# Section 2: Table of Contents
# ============================================
def table_of_contents(sections):
    data = [["Section", "Description"]]
    for s in sections:
        data.append([s[0], s[1]])
    return [
        Paragraph("📖 Table of Contents", section_style),
        Spacer(1, 6),
        zebra_table(data, col_widths=[100 * mm, 80 * mm]),
        PageBreak(),
    ]

# ============================================
# Section 3: Metric Content Blocks
# ============================================
def metric_section(title, table_data=None, insights=None):
    elements = [Paragraph(f"📊 {title}", section_style), Spacer(1, 8)]
    if table_data is not None:
        elements.append(zebra_table(table_data))
        elements.append(Spacer(1, 8))
    if insights:
        elements.append(Paragraph("💡 Key Insights:", summary_style))
        for insight in insights:
            elements.append(Paragraph(f"• {insight}", base_style))
        elements.append(Spacer(1, 12))
    elements.append(PageBreak())
    return elements

# ============================================
# Section 4: Consolidated Insights Summary
# ============================================
def summary_section(all_insights):
    data = [["Metric", "Insight Summary"]]
    for k, v in all_insights.items():
        data.append([k, v])
    return [
        Paragraph("🧾 Consolidated Insights Summary", section_style),
        Spacer(1, 8),
        zebra_table(data, col_widths=[60 * mm, 120 * mm]),
        PageBreak(),
    ]

# ============================================
# Main Export Function
# ============================================
def generate_pdf_report(report_title, module_name, sections, insights, author="Amlan Mishra"):
    """
    Creates full executive PDF with cover, TOC, metrics, and insights.
    sections = [(title, description, table_data, insight_list), ...]
    insights = {metric: summary}
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    elements = []
    # Cover Page
    elements += cover_page(report_title, author, module_name)
    # TOC
    toc_data = [(s[0], s[1]) for s in sections]
    elements += table_of_contents(toc_data)
    # Each Metric Section
    for title, desc, table_data, insight_list in sections:
        elements += metric_section(title, table_data, insight_list)
    # Consolidated Summary
    elements += summary_section(insights)

    # Footer & Branding
    elements.append(Paragraph(
        "<para align=center><font size=9 color='#6B7280'>Prepared with ❤️ by Amlan Mishra | © 2025 HR Tech Portfolio</font></para>",
        base_style
    ))

    doc.build(elements)
    return buffer.getvalue()

# ============================================
# Streamlit Wrapper for Download Button
# ============================================
import streamlit as st

def render_pdf_download_button(report_title, html_summary, filename_prefix, module_name="Module Report"):
    """
    Renders a PDF download button for Streamlit using the new Executive PDF layout.
    html_summary = optional simple summary text for compatibility.
    """
    try:
        st.subheader("📄 Export Executive Report (PDF)")
        fake_sections = [
            ("Overview", "Executive overview and context", [["Metric", "Value"], ["Performance Index", "78%"]], ["Sample insight"]),
            ("Analysis", "Detailed performance metrics", [["Rating", "CTC"], ["5", "15.0 LPA"], ["4", "12.2 LPA"]], ["Top performers align with high CTC"]),
        ]
        fake_insights = {"Performance": "High correlation between rating and CTC"}
        pdf_data = generate_pdf_report(report_title, module_name, fake_sections, fake_insights)
        st.download_button(
            label="⬇️ Download Executive Report (PDF)",
            data=pdf_data,
            file_name=f"{filename_prefix}_Executive_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"⚠️ Failed to generate report: {e}")