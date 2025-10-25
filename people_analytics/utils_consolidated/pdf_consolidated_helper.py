# ============================================
# utils_consolidated/pdf_consolidated_helper.py — v5.4 | Kage Stable (Insights + Finale)
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
# 🧩 Font Setup
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
HEADER_COLOR = colors.HexColor("#0F172A")
TABLE_HEADER_BG = colors.HexColor("#E5E7EB")
BODY_TEXT = colors.HexColor("#111827")

# -----------------------------------------------------------
# 📄 Footer
# -----------------------------------------------------------
def _add_footer(canvas, doc):
    canvas.saveState()
    footer_text = "Prepared with ❤️ by Amlan Mishra — People Analytics Project (2025)"
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0] / 2, 15, footer_text)
    canvas.restoreState()

# -----------------------------------------------------------
# 📘 Build Consolidated PDF
# -----------------------------------------------------------
def render_consolidated_pdf(report_title: str, modules_payload: list, filename_prefix: str):
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
            heading = ParagraphStyle("Heading", fontName=FONT_NAME, fontSize=14, textColor=PRIMARY_COLOR)
            body = ParagraphStyle("Body", fontName=FONT_NAME, fontSize=10, textColor=BODY_TEXT)
            title = ParagraphStyle("Title", fontName=FONT_NAME, fontSize=22, textColor=HEADER_COLOR, alignment=1)
            sub = ParagraphStyle("Sub", fontName=FONT_NAME, fontSize=12, textColor=colors.HexColor("#374151"), alignment=1)

            story = []

            # Cover Page
            story.append(Spacer(1, 70))
            story.append(Paragraph(f"<b>{report_title}</b>", title))
            story.append(Spacer(1, 12))
            story.append(Paragraph("People Analytics — Leadership Insights Deck", sub))
            story.append(Spacer(1, 50))
            story.append(Paragraph(f"<font size=10>Generated on {datetime.now().strftime('%d %b %Y, %H:%M %p')}</font>", sub))
            story.append(PageBreak())

            # Table of Contents
            toc_data = [["#", "Module", "Description", "Page"]]
            for i, mod in enumerate(modules_payload, 1):
                toc_data.append([i, mod.get("module_name", ""), mod.get("module_desc", ""), str(i + 1)])
            toc_table = Table(toc_data, colWidths=[25, 100, 250, 35])
            toc_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 9)
            ]))
            story.append(Paragraph("<b>Table of Contents</b>", heading))
            story.append(Spacer(1, 8))
            story.append(toc_table)
            story.append(PageBreak())

            # Modules
            for mod in modules_payload:
                module_name = mod.get("module_name", "Unknown")
                module_desc = mod.get("module_desc", "")
                data_blocks = mod.get("data_blocks", [])

                story.append(Spacer(1, 30))
                story.append(Paragraph(f"<para align=center><font size=18 color='white'><b>{module_name}</b></font></para>",
                    ParagraphStyle("Divider", backColor=PRIMARY_COLOR, alignment=1, spaceBefore=40, spaceAfter=40)))
                story.append(Spacer(1, 15))
                story.append(Paragraph(f"<para align=center><font size=10 color='#6B7280'>{module_desc}</font></para>", body))
                story.append(PageBreak())

                for block in data_blocks:
                    title_txt = block.get("title", "")
                    desc = block.get("desc", "")
                    df = block.get("df")
                    fig = block.get("fig")
                    insights = block.get("insights", [])

                    story.append(Paragraph(title_txt, heading))
                    story.append(Paragraph(desc, body))
                    story.append(Spacer(1, 6))

                    if df is not None and not df.empty:
                        df = df.round(2).astype(str)
                        data = [list(df.columns)] + df.values.tolist()
                        table = Table(data, colWidths=[(A4[0]-50)/len(df.columns)]*len(df.columns))
                        table.setStyle(TableStyle([
                            ("GRID",(0,0),(-1,-1),0.25,colors.black),
                            ("BACKGROUND",(0,0),(-1,0),TABLE_HEADER_BG),
                            ("FONTNAME",(0,0),(-1,-1),FONT_NAME),
                            ("FONTSIZE",(0,0),(-1,-1),9)
                        ]))
                        story.append(table)
                        story.append(Spacer(1,8))

                    if fig is not None:
                        img_path = ensure_chart_saved(title_txt, fig)
                        if img_path and os.path.exists(img_path):
                            story.append(RLImage(img_path, width=170*mm, height=95*mm))
                            story.append(Spacer(1,8))

                    if insights:
                        joined = " • ".join(insights)
                        story.append(Paragraph(f"<font color='#2563EB'><i>{joined}</i></font>", body))
                    story.append(PageBreak())

            # Consolidated Insights Summary
            story.append(Paragraph("Consolidated Insights Summary", heading))
            story.append(Spacer(1, 10))
            for mod in modules_payload:
                module_name = mod.get("module_name", "Unknown Module")
                story.append(Paragraph(f"<b>{module_name}</b>", heading))
                insights_table_data = [["Section", "Key Insights"]]
                for block in mod.get("data_blocks", []):
                    insights_table_data.append([block.get("title",""), " • ".join(block.get("insights",[])) or "—"])
                insights_table = Table(insights_table_data, colWidths=[120, 330])
                insights_table.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),TABLE_HEADER_BG),
                    ("GRID",(0,0),(-1,-1),0.25,colors.black),
                    ("FONTNAME",(0,0),(-1,-1),FONT_NAME),
                    ("FONTSIZE",(0,0),(-1,-1),9),
                    ("VALIGN",(0,0),(-1,-1),"TOP")
                ]))
                story.append(insights_table)
                story.append(Spacer(1, 12))
            story.append(PageBreak())

            # Thank You Page
            story.append(Spacer(1,150))
            story.append(Paragraph(
                "<para align=center><font size=28 color='white'><b>THANK YOU</b></font></para>",
                ParagraphStyle("ThankYou", backColor=PRIMARY_COLOR, alignment=1, leading=30)
            ))
            story.append(Spacer(1,20))
            story.append(Paragraph(
                "<para align=center><font size=12 color='white'>Prepared with ❤️ by Amlan Mishra | People Analytics Project</font></para>",
                ParagraphStyle("ThankYouSub", backColor=PRIMARY_COLOR, alignment=1)
            ))

            doc.build(story, onLaterPages=_add_footer)
            pdf_bytes = buf.getvalue()
            st.success("✅ Consolidated Leadership Deck generated successfully!")
            st.download_button(
                "⬇️ Download HR Leadership Deck (PDF)",
                pdf_bytes,
                file_name=f"{filename_prefix}_Leadership_Deck.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"⚠️ PDF generation failed: {e}")