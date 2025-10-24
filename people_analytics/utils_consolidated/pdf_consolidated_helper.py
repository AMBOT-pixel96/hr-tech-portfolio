# ============================================
# utils_consolidated/pdf_consolidated_helper.py — v6.3 | Boardroom Diagnostic Build
# ============================================
import os, io, traceback
import streamlit as st
from datetime import datetime
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
# 🧩 Font Setup
# -----------------------------------------------------------
try:
    pdfmetrics.registerFont(
        TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    FONT_NAME = "DejaVuSans"
except:
    FONT_NAME = "Helvetica"

# -----------------------------------------------------------
# 🎨 Styling
# -----------------------------------------------------------
PRIMARY_COLOR = colors.HexColor("#1E3A8A")
ACCENT_COLOR = colors.HexColor("#2563EB")
HEADER_COLOR = colors.HexColor("#0F172A")
TABLE_HEADER_BG = colors.HexColor("#E5E7EB")
BODY_TEXT = colors.HexColor("#111827")

# -----------------------------------------------------------
# 📄 Footer
# -----------------------------------------------------------
def _add_footer(canvas, doc):
    canvas.saveState()
    footer_text = "Prepared with ❤️ by People Analytics Project — 2025"
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0] / 2, 15, footer_text)
    canvas.restoreState()

# -----------------------------------------------------------
# 🧾 PDF Generator
# -----------------------------------------------------------
def render_consolidated_pdf(report_title: str, modules_payload: list, filename_prefix: str):
    if not modules_payload:
        st.warning("⚠️ No module data available for PDF generation.")
        return

    if st.button("🧾 Generate Consolidated Executive Deck", use_container_width=True):
        try:
            st.info("🧠 Generating PDF — this may take a few seconds...")

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
                "Title", fontName=FONT_NAME, fontSize=22, alignment=1,
                textColor=HEADER_COLOR, leading=26
            )
            subtitle_style = ParagraphStyle(
                "Subtitle", fontName=FONT_NAME, fontSize=13, alignment=1,
                textColor=colors.HexColor("#374151")
            )
            heading = ParagraphStyle(
                "Heading", fontName=FONT_NAME, fontSize=14,
                textColor=PRIMARY_COLOR, spaceAfter=6
            )
            body = ParagraphStyle(
                "Body", fontName=FONT_NAME, fontSize=10,
                textColor=BODY_TEXT, leading=13
            )

            story = []

            # ------------------ COVER ------------------
            story.append(Spacer(1, 70))
            story.append(Paragraph(f"<b>{report_title}</b>", title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph("People Analytics — Leadership Insights Deck", subtitle_style))
            story.append(Spacer(1, 40))
            story.append(Paragraph(
                f"<font size=10>Generated on {datetime.now().strftime('%d %b %Y, %H:%M %p')}</font>",
                subtitle_style,
            ))
            story.append(PageBreak())

            # ------------------ MODULES ------------------
            executive_summary = [["Module", "Key Insights"]]

            for mod in modules_payload:
                module_name = mod.get("module_name", "Unknown Module")
                st.write(f"📘 Processing module: {module_name}")
                module_desc = mod.get("module_desc", "")
                data_blocks = mod.get("data_blocks", [])

                # Divider Page
                story.append(Paragraph(f"<para align=center><b>{module_name.upper()}</b></para>", heading))
                story.append(Paragraph(module_desc, body))
                story.append(PageBreak())

                for block in data_blocks:
                    title = block.get("title", "")
                    desc = block.get("desc", "")
                    df = block.get("df", None)
                    fig = block.get("fig", None)
                    insights = block.get("insights", [])

                    story.append(Paragraph(f"<b>{title}</b>", heading))
                    story.append(Paragraph(desc, body))
                    story.append(Spacer(1, 6))

                    # Data Table (defensive)
                    if df is not None and not df.empty:
                        try:
                            df = df.copy().fillna("").astype(str)
                            table_data = [list(df.columns)] + df.values.tolist()
                            table = Table(table_data, colWidths=[(A4[0] - 40) / len(df.columns)] * len(df.columns))
                            table.setStyle(TableStyle([
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                                ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                                ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ]))
                            story.append(table)
                        except Exception as e:
                            st.warning(f"⚠️ Skipped a table in {title}: {e}")

                    # Chart
                    if fig is not None:
                        try:
                            img_path = ensure_chart_saved(title, fig)
                            if img_path and os.path.exists(img_path):
                                story.append(RLImage(img_path, width=170 * mm, height=95 * mm))
                            else:
                                st.warning(f"⚠️ Chart not rendered for {title}")
                        except Exception as e:
                            st.warning(f"⚠️ Chart export failed: {e}")

                    # Insights
                    if insights:
                        joined = " • ".join(str(i) for i in insights)
                        story.append(Paragraph(f"<font color='{ACCENT_COLOR}'><i>{joined}</i></font>", body))
                        executive_summary.append([module_name, joined])

                    story.append(PageBreak())

            # ------------------ SUMMARY ------------------
            story.append(Paragraph("<b>Executive Summary</b>", heading))
            story.append(Spacer(1, 10))
            summary_table = Table(executive_summary, colWidths=[140, 310])
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(summary_table)

            # ------------------ BUILD ------------------
            doc.build(story, onLaterPages=_add_footer)
            pdf_bytes = buf.getvalue()

            if pdf_bytes:
                st.success("✅ PDF generated successfully!")
                st.download_button(
                    "⬇️ Download HR Leadership Deck (PDF)",
                    pdf_bytes,
                    file_name=f"{filename_prefix}_Leadership_Deck.pdf",
                    mime="application/pdf",
                )
            else:
                st.error("⚠️ PDF generation returned empty bytes!")

        except Exception as e:
            st.error("🚨 PDF generation failed — details below:")
            st.code(traceback.format_exc())