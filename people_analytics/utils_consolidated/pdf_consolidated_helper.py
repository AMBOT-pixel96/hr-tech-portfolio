# ============================================
# utils_consolidated/pdf_consolidated_helper.py
# v5.3 | Cloud-Stable ReportLab PDF Builder
# ============================================

import os
import io
from datetime import datetime
import streamlit as st

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from utils_consolidated.chart_consolidated_saver import ensure_chart_saved


# -----------------------------------------------------------
# 🧩 Font Setup (for ₹, %, etc.)
# -----------------------------------------------------------
try:
    pdfmetrics.registerFont(
        TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    FONT_NAME = "DejaVuSans"
except Exception:
    FONT_NAME = "Helvetica"

# -----------------------------------------------------------
# 🎨 Global Colors
# -----------------------------------------------------------
PRIMARY_COLOR = colors.HexColor("#1E3A8A")
ACCENT_COLOR = colors.HexColor("#2563EB")
HEADER_COLOR = colors.HexColor("#0F172A")
TABLE_HEADER_BG = colors.HexColor("#E5E7EB")
TABLE_HEADER_TEXT = colors.black
BODY_TEXT = colors.HexColor("#111827")

# -----------------------------------------------------------
# 📄 Add Footer to every page
# -----------------------------------------------------------
def _add_footer(canvas, doc):
    canvas.saveState()
    footer_text = "Prepared with ❤️ by People Analytics Project — 2025"
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0] / 2, 15, footer_text)
    canvas.restoreState()

# -----------------------------------------------------------
# 📘 Report Generator (pure ReportLab)
# -----------------------------------------------------------
def render_consolidated_pdf(report_title: str, modules_payload: list, filename_prefix: str):
    """
    Builds a full consolidated executive report with:
      🧠 Cover Page
      📖 Table of Contents
      📊 Module Sections
      🧾 Executive Summary
    """
    if not modules_payload:
        st.warning("⚠️ No module data available for PDF generation.")
        return

    if st.button("🧾 Generate Consolidated Executive Deck", use_container_width=True):
        try:
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
            title_style = ParagraphStyle(
                "Title",
                fontName=FONT_NAME,
                fontSize=22,
                alignment=1,
                textColor=HEADER_COLOR,
                leading=26,
            )
            subtitle_style = ParagraphStyle(
                "Subtitle",
                fontName=FONT_NAME,
                fontSize=13,
                alignment=1,
                textColor=colors.HexColor("#374151"),
            )
            small_grey = ParagraphStyle(
                "SmallGrey",
                fontName=FONT_NAME,
                fontSize=9,
                textColor=colors.HexColor("#6B7280"),
            )
            heading = ParagraphStyle(
                "Heading",
                fontName=FONT_NAME,
                fontSize=14,
                textColor=PRIMARY_COLOR,
                spaceAfter=6,
            )
            body = ParagraphStyle(
                "Body",
                fontName=FONT_NAME,
                fontSize=10,
                textColor=BODY_TEXT,
                leading=13,
            )

            story = []

            # ---------------------------------------------------
            # 🧠 COVER PAGE
            # ---------------------------------------------------
            story.append(Spacer(1, 70))
            story.append(Paragraph(f"<b>{report_title}</b>", title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph("People Analytics — Leadership Insights Deck", subtitle_style))
            story.append(Spacer(1, 40))
            story.append(Paragraph(
                f"<font size=10>Generated on {datetime.now().strftime('%d %b %Y, %H:%M %p')}</font>",
                subtitle_style,
            ))
            story.append(Spacer(1, 60))
            story.append(Paragraph(
                "<para align=center><font size=9 color='#6B7280'>© 2025 People Analytics Project | Confidential</font></para>",
                small_grey,
            ))
            story.append(PageBreak())

            # ---------------------------------------------------
            # 📖 TABLE OF CONTENTS
            # ---------------------------------------------------
            toc_data = [["#", "Module", "Description", "Page"]]
            for i, mod in enumerate(modules_payload, 1):
                toc_data.append([
                    i,
                    mod.get("module_name", ""),
                    mod.get("module_desc", ""),
                    str(i + 1),
                ])
            toc_table = Table(toc_data, colWidths=[25, 110, 240, 35])
            toc_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), TABLE_HEADER_TEXT),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]))
            story.append(Paragraph("<b>Table of Contents</b>", heading))
            story.append(Spacer(1, 8))
            story.append(toc_table)
            story.append(PageBreak())

            # ---------------------------------------------------
            # 📊 MODULE SECTIONS
            # ---------------------------------------------------
            executive_summary = [["Module", "Key Insights"]]

            for mod in modules_payload:
                module_name = mod.get("module_name", "Unknown Module")
                module_desc = mod.get("module_desc", "")
                data_blocks = mod.get("data_blocks", [])

                # Divider Page
                story.append(Spacer(1, 40))
                story.append(Paragraph(
                    f"<para align=center><font size=16 color='white'><b>{module_name.upper()}</b></font></para>",
                    ParagraphStyle("Divider", backColor=PRIMARY_COLOR, alignment=1, spaceBefore=40, spaceAfter=40, leading=24)
                ))
                story.append(Spacer(1, 20))
                story.append(Paragraph(
                    f"<para align=center><font size=10 color='#6B7280'>{module_desc}</font></para>",
                    small_grey,
                ))
                story.append(PageBreak())

                for block in data_blocks:
                    title = block.get("title", "")
                    desc = block.get("desc", "")
                    df = block.get("df", None)
                    fig = block.get("fig", None)
                    insights = block.get("insights", [])

                    story.append(Paragraph(f"{title}", heading))
                    story.append(Paragraph(desc, body))
                    story.append(Spacer(1, 8))

                    if df is not None and not df.empty:
                        df = df.round(2).astype(str)
                        table_data = [list(df.columns)] + df.values.tolist()
                        table = Table(
                            table_data,
                            colWidths=[(A4[0] - 40) / len(df.columns)] * len(df.columns),
                        )
                        table.setStyle(TableStyle([
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                            ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                            ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 10))

                    if fig is not None:
                        img_path = ensure_chart_saved(title, fig)
                        if img_path and os.path.exists(img_path):
                            story.append(RLImage(img_path, width=170 * mm, height=95 * mm))
                            story.append(Spacer(1, 8))
                        else:
                            story.append(Paragraph("⚠️ Chart could not be rendered.", body))

                    if insights:
                        joined = " • ".join(str(i) for i in insights)
                        story.append(Paragraph(f"<font color='{ACCENT_COLOR}'><i>{joined}</i></font>", body))
                        executive_summary.append([module_name, joined])
                    else:
                        executive_summary.append([module_name, "No explicit insights."])

                    story.append(PageBreak())

            # ---------------------------------------------------
            # 🧾 EXECUTIVE SUMMARY
            # ---------------------------------------------------
            story.append(Paragraph("Executive Summary", heading))
            story.append(Spacer(1, 10))
            summary_table = Table(executive_summary, colWidths=[140, 310])
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(summary_table)

            # ---------------------------------------------------
            # 💾 BUILD & DOWNLOAD
            # ---------------------------------------------------
            doc.build(story, onLaterPages=_add_footer)
            pdf_bytes = buf.getvalue()
            st.success("✅ Consolidated Leadership Deck generated successfully!")
            st.download_button(
                "⬇️ Download HR Leadership Deck (PDF)",
                pdf_bytes,
                file_name=f"{filename_prefix}_Leadership_Deck.pdf",
                mime="application/pdf",
            )

        except Exception as e:
            st.error(f"⚠️ PDF generation failed: {e}")
            st.stop()