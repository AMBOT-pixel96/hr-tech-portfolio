# ============================================
# utils/pdf_helper.py — v3 Executive Production
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
import streamlit as st

# ----------------------------------------------------
# Font registration (for ₹, emojis, special characters)
# ----------------------------------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
except:
    pass

# ----------------------------------------------------
# Styles
# ----------------------------------------------------
styles = getSampleStyleSheet()
base_style = ParagraphStyle(
    "Base",
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
    textColor=colors.HexColor("#111827"),
)

# ----------------------------------------------------
# Helper: Zebra Table
# ----------------------------------------------------
def zebra_table(data, col_widths=None):
    """Creates zebra-styled tables."""
    if not data:
        return Paragraph("No data available.", base_style)

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
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

# ----------------------------------------------------
# Section 1: Cover Page
# ----------------------------------------------------
def cover_page(report_title, module_name, author="Amlan Mishra"):
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    return [
        Spacer(1, 120),
        Paragraph(report_title, title_style),
        Spacer(1, 8),
        Paragraph(f"<b>{module_name}</b>", subtitle_style),
        Spacer(1, 200),
        Paragraph(f"Prepared with ❤️ by <b>{author}</b>", base_style),
        Paragraph(f"Generated on {now}", base_style),
        PageBreak(),
    ]

# ----------------------------------------------------
# Section 2: Table of Contents
# ----------------------------------------------------
def table_of_contents(section_titles):
    data = [["Section", "Description"]]
    for title, desc in section_titles:
        data.append([title, desc])
    return [
        Paragraph("📖 Table of Contents", section_style),
        Spacer(1, 6),
        zebra_table(data, col_widths=[80 * mm, 90 * mm]),
        PageBreak(),
    ]

# ----------------------------------------------------
# Section 3: Metric Sections
# ----------------------------------------------------
def metric_section(title, description, table_data=None, insights=None):
    elements = [
        Paragraph(f"📊 {title}", section_style),
        Paragraph(description, summary_style),
        Spacer(1, 8)
    ]

    if table_data is not None:
        elements.append(zebra_table(table_data))
        elements.append(Spacer(1, 8))

    if insights:
        elements.append(Paragraph("💡 Key Insights:", summary_style))
        for i in insights:
            elements.append(Paragraph(f"• {i}", base_style))
        elements.append(Spacer(1, 10))

    elements.append(PageBreak())
    return elements

# ----------------------------------------------------
# Section 4: Consolidated Insights Summary
# ----------------------------------------------------
def summary_section(all_insights):
    data = [["Metric", "Insight Summary"]]
    for metric, insight in all_insights.items():
        data.append([metric, insight])
    return [
        Paragraph("🧾 Consolidated Insights Summary", section_style),
        Spacer(1, 6),
        zebra_table(data, col_widths=[60 * mm, 120 * mm]),
        PageBreak(),
    ]

# ----------------------------------------------------
# Section 5: Full PDF Generator
# ----------------------------------------------------
def generate_pdf_report(report_title, module_name, sections, all_insights, author="Amlan Mishra"):
    """
    Generates a full executive PDF report.
    sections = [
        {"title": "Metric Name", "desc": "Description", "table": [[...]], "insights": ["...", "..."]},
        ...
    ]
    all_insights = {metric_name: summary_text, ...}
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
    elements += cover_page(report_title, module_name, author)
    # Table of Contents
    toc_data = [(s["title"], s.get("desc", "")) for s in sections]
    elements += table_of_contents(toc_data)
    # Metric Sections
    for s in sections:
        elements += metric_section(s["title"], s.get("desc", ""), s.get("table"), s.get("insights"))
    # Summary
    elements += summary_section(all_insights)
    # Footer
    elements.append(Paragraph(
        "<para align=center><font size=9 color='#6B7280'>Prepared with ❤️ by Amlan Mishra | © 2025 HR Tech Portfolio</font></para>",
        base_style,
    ))

    doc.build(elements)
    return buffer.getvalue()

# ----------------------------------------------------
# Streamlit Wrapper — Plug & Play
# ----------------------------------------------------
def render_pdf_download_button(report_title, module_name, sections, all_insights, filename_prefix):
    """
    Renders a fully functional PDF download button for any module.
    Example call:
        render_pdf_download_button(
            report_title="Engagement Analytics",
            module_name="Engagement",
            sections=sections_list,
            all_insights=insight_dict,
            filename_prefix="Engagement_Report"
        )
    """
    try:
        st.subheader("📄 Export Executive Report (PDF)")
        pdf_bytes = generate_pdf_report(report_title, module_name, sections, all_insights)
        st.download_button(
            label="⬇️ Download Executive Report (PDF)",
            data=pdf_bytes,
            file_name=f"{filename_prefix}_Executive_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"⚠️ Error generating PDF: {e}")