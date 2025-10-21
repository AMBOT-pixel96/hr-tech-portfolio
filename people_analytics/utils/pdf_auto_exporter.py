# ============================================
# utils/pdf_auto_exporter.py — v3.0 | Executive Layout Edition
# ============================================

from io import BytesIO
import os, datetime, textwrap
import pandas as pd
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Image as RLImage, Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import plotly.io as pio

# ---------------------------
# 🧩 Kaleido Init
# ---------------------------
try:
    pio.renderers.default = "kaleido"
except Exception as e:
    print(f"⚠️ Kaleido init failed: {e}")

# ---------------------------
# 🔤 Font Setup
# ---------------------------
DEFAULT_FONT_NAME = "DejaVuSans"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_PATH))
except Exception as e:
    print(f"⚠️ Font registration skipped: {e}")

# ---------------------------
# 📊 Helpers
# ---------------------------
def fig_to_png_bytes(fig, width=900, height=520, scale=1):
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception as e:
        print(f"⚠️ Kaleido export failed: {e}")
        return None

def _df_to_table_data(df: pd.DataFrame, max_rows=20):
    if df is None or df.empty:
        return [["No data available."]]
    df2 = df.head(max_rows).copy().fillna("").astype(str)
    return [list(df2.columns)] + df2.values.tolist()

def _zebra_style(cols, rows):
    style = TableStyle()
    style.add("FONTNAME", (0,0), (-1,-1), DEFAULT_FONT_NAME)
    style.add("FONTSIZE", (0,0), (-1,-1), 9)
    style.add("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1E3A8A"))
    style.add("TEXTCOLOR", (0,0), (-1,0), colors.white)
    for r in range(1, rows):
        bg = colors.HexColor("#F9FAFB") if r % 2 == 0 else colors.white
        style.add("BACKGROUND", (0,r), (-1,r), bg)
    style.add("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CBD5E1"))
    style.add("VALIGN", (0,0), (-1,-1), "MIDDLE")
    return style

# ---------------------------
# 🧾 Executive PDF Builder
# ---------------------------
def export_module_report(report_title: str, module_name: str, data_blocks: list, filename_prefix: str = None) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm, topMargin=25*mm, bottomMargin=20*mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                 fontName=DEFAULT_FONT_NAME, fontSize=22,
                                 alignment=1, textColor=colors.HexColor("#0F172A"))
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                    fontName=DEFAULT_FONT_NAME, fontSize=12,
                                    alignment=1, textColor=colors.HexColor("#374151"))
    body = ParagraphStyle("Body", parent=styles["Normal"],
                          fontName=DEFAULT_FONT_NAME, fontSize=10,
                          leading=13, textColor=colors.black)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontName=DEFAULT_FONT_NAME, fontSize=14,
                        textColor=colors.HexColor("#111827"))

    story = []

    # --- Cover ---
    story.append(Spacer(1, 100))
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"{module_name} Module", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Generated on {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", subtitle_style))
    story.append(Spacer(1, 200))
    story.append(Paragraph("<b>Prepared by:</b> Amlan Mishra", subtitle_style))
    story.append(PageBreak())

    # --- TOC ---
    toc_rows = [["#", "Section", "Description", "Page"]]
    for i, block in enumerate(data_blocks, 1):
        desc = block.get("desc", "")
        if len(desc) > 70: desc = desc[:67] + "..."
        toc_rows.append([str(i), block.get("title", f"Section {i}"), desc, str(i+2)])
    toc_table = Table(toc_rows, colWidths=[15*mm, 60*mm, 80*mm, 15*mm])
    toc_table.setStyle(_zebra_style(4, len(toc_rows)))
    story.append(Paragraph("Table of Contents", h2))
    story.append(Spacer(1,6))
    story.append(toc_table)
    story.append(PageBreak())

    # --- Sections ---
    summary = []
    for i, block in enumerate(data_blocks, 1):
        title = block.get("title", f"Section {i}")
        desc = block.get("desc", "")
        df, fig, insights = block.get("df"), block.get("fig"), block.get("insights", [])

        story.append(Paragraph(f"{i}. {title}", h2))
        story.append(Paragraph(desc, body))
        story.append(Spacer(1,8))

        # Metric Table
        if df is not None:
            table_data = _df_to_table_data(df, max_rows=20)
            table = Table(table_data, repeatRows=1, hAlign="LEFT")
            table.setStyle(_zebra_style(len(table_data[0]), len(table_data)))
            story.append(table)
            story.append(Spacer(1,8))

        # Graph
        if fig is not None:
            img = fig_to_png_bytes(fig)
            if img:
                story.append(RLImage(BytesIO(img), width=160*mm, height=90*mm))
                story.append(Spacer(1,8))
            else:
                story.append(Paragraph("⚠️ Graph rendering failed (Kaleido).", body))

        # Insights
        if insights:
            bullets = "<br/>".join([f"• {textwrap.shorten(i, width=110)}" for i in insights])
            story.append(Paragraph(bullets, body))
            summary.append([title, " ; ".join(insights)])

        story.append(PageBreak())

    # --- Summary ---
    story.append(Paragraph("Executive Summary", h2))
    summary = summary or [["—","No insights recorded."]]
    table = Table([["Section","Key Insights"]]+summary, colWidths=[60*mm,100*mm])
    table.setStyle(_zebra_style(2,len(summary)+1))
    story.append(table)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf