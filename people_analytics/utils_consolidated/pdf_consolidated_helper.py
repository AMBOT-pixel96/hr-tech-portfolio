# ============================================
# utils_consolidated/pdf_consolidated_helper.py — v6.4 | Fail-Safe Build
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

try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    FONT_NAME = "DejaVuSans"
except:
    FONT_NAME = "Helvetica"

def _add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0] / 2, 15, "Prepared with ❤️ by People Analytics Project — 2025")
    canvas.restoreState()

def render_consolidated_pdf(report_title: str, modules_payload: list, filename_prefix: str):
    if not modules_payload:
        st.warning("⚠️ No module data available for PDF generation.")
        return

    if st.button("🧾 Generate Consolidated Executive Deck", use_container_width=True):
        try:
            st.info("🧠 Generating PDF... please wait (15–20 seconds)")

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
            heading = ParagraphStyle("Heading", fontName=FONT_NAME, fontSize=14, textColor=colors.HexColor("#1E3A8A"), spaceAfter=6)
            body = ParagraphStyle("Body", fontName=FONT_NAME, fontSize=10, textColor=colors.HexColor("#111827"), leading=13)

            story = []
            story.append(Spacer(1, 60))
            story.append(Paragraph(f"<b>{report_title}</b>", heading))
            story.append(Paragraph(datetime.now().strftime("%d %B %Y, %H:%M %p"), body))
            story.append(PageBreak())

            for mod in modules_payload:
                module_name = mod.get("module_name", "Unknown Module")
                module_desc = mod.get("module_desc", "")
                data_blocks = mod.get("data_blocks", [])
                story.append(Paragraph(f"<b>{module_name}</b>", heading))
                story.append(Paragraph(module_desc, body))
                story.append(PageBreak())

                for block in data_blocks:
                    title = block.get("title", "")
                    df = block.get("df", None)
                    fig = block.get("fig", None)
                    insights = block.get("insights", [])
                    story.append(Paragraph(f"<b>{title}</b>", heading))

                    # Table safe rendering
                    try:
                        if df is not None and not df.empty:
                            df = df.copy().fillna("").astype(str)
                            table_data = [list(df.columns)] + df.values.tolist()
                            table = Table(table_data)
                            table.setStyle(TableStyle([
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                                ("FONTSIZE", (0, 0), (-1, -1), 9)
                            ]))
                            story.append(table)
                    except Exception as e:
                        st.warning(f"⚠️ Table skipped for {title}: {e}")

                    # Chart safe rendering
                    try:
                        if fig is not None:
                            img_path = ensure_chart_saved(title, fig)
                            if img_path and os.path.getsize(img_path) > 500:
                                story.append(RLImage(img_path, width=170 * mm, height=95 * mm))
                            else:
                                st.warning(f"⚠️ Chart skipped for {title}")
                    except Exception as e:
                        st.warning(f"⚠️ Chart render failed: {e}")

                    # Insights
                    if insights:
                        story.append(Paragraph(" • ".join(map(str, insights)), body))
                    story.append(PageBreak())

            doc.build(story, onLaterPages=_add_footer)
            pdf_bytes = buf.getvalue()

            if pdf_bytes:
                st.success("✅ PDF generated successfully!")
                st.download_button("⬇️ Download HR Leadership Deck (PDF)", pdf_bytes, file_name=f"{filename_prefix}_Leadership_Deck.pdf", mime="application/pdf")
            else:
                st.error("⚠️ Empty PDF output.")

        except Exception as e:
            st.error("🚨 PDF generation failed:")
            st.code(traceback.format_exc())