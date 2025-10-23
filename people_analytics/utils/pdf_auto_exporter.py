# ============================================
# utils/pdf_auto_exporter.py — v3.7 | File-based Renderer Edition
# ============================================

from io import BytesIO
import os, datetime, textwrap
import pandas as pd
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --------------------------------------------
# 🧩 Font Setup
# --------------------------------------------
DEFAULT_FONT_NAME = "DejaVuSans"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    pdfmetrics.registerFont(TTFont(DEFAULT_FONT_NAME, FONT_PATH))
except Exception as e:
    print(f"⚠️ Font registration skipped: {e}")
    DEFAULT_FONT_NAME = "Helvetica"

# --------------------------------------------
# 🧠 Styling
# --------------------------------------------
styles = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontName=DEFAULT_FONT_NAME, fontSize=10, leading=13)
H2 = ParagraphStyle("Heading2", parent=styles["Heading2"], fontName=DEFAULT_FONT_NAME, fontSize=14, textColor=colors.HexColor("#111827"))
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontName=DEFAULT_FONT_NAME, fontSize=20, alignment=1)

# --------------------------------------------
# 📊 Table Helpers
# --------------------------------------------
def _format_val(v):
    try:
        if isinstance(v, (float, int)):
            return f"{v:.2f}" if not float(v).is_integer() else str(int(v))
        return str(v)
    except Exception:
        return str(v)

def _df_to_table_data(df: pd.DataFrame, max_rows=15, max_cols=6):
    if df is None or df.empty:
        return [["No data available."]]
    df2 = df.head(max_rows).copy()
    if df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]
    df2 = df2.fillna("").applymap(_format_val)
    header = list(df2.columns)
    rows = df2.values.tolist()
    return [header] + rows

def _zebra_table_style(n_cols, n_rows):
    style = TableStyle()
    style.add("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT_NAME)
    style.add("FONTSIZE", (0, 0), (-1, -1), 9)
    style.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A"))
    style.add("TEXTCOLOR", (0, 0), (-1, 0), colors.white)
    for r in range(1, n_rows):
        bg = colors.HexColor("#F7F7F7") if r % 2 else colors.white
        style.add("BACKGROUND", (0, r), (-1, r), bg)
    style.add("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB"))
    return style

# --------------------------------------------
# 🧾 Main Export Function
# --------------------------------------------
def export_module_report(report_title: str, module_name: str, data_blocks: list, filename_prefix: str = None) -> bytes:
    """
    Generate a complete PDF report using pre-saved chart images from tmp_charts/.
    Each data_block can include {'title', 'desc', 'df', 'fig_path', 'insights'}.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=22*mm, bottomMargin=18*mm)
    story = []

    # === Cover Page ===
    story.append(Spacer(1, 70))
    story.append(Paragraph(report_title, TITLE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"{module_name} Module", BODY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", BODY))
    story.append(PageBreak())

    # === Table of Contents ===
    toc_rows = [["#", "Section", "Description", "Page"]]
    for i, block in enumerate(data_blocks, 1):
        desc = block.get("desc", "")
        if len(desc) > 70:
            desc = desc[:67] + "..."
        toc_rows.append([str(i), block.get("title", f"Section {i}"), desc, str(i + 1)])
    toc_table = Table(toc_rows, colWidths=[15*mm, 65*mm, 85*mm, 15*mm])
    toc_table.setStyle(_zebra_table_style(4, len(toc_rows)))
    story.append(Paragraph("Table of Contents", H2))
    story.append(Spacer(1, 6))
    story.append(toc_table)
    story.append(PageBreak())

    # === Main Sections ===
    summary_insights = []
    for i, block in enumerate(data_blocks, 1):
        story.append(Paragraph(f"{i}. {block.get('title', f'Section {i}')}", H2))
        desc = block.get("desc", "")
        if desc:
            story.append(Paragraph(desc, BODY))
        story.append(Spacer(1, 8))

        # Table
        df = block.get("df")
        if df is not None and not df.empty:
            table_data = _df_to_table_data(df)
            n_cols = len(table_data[0])
            col_widths = [max(30, min(55, 500 / n_cols)) * mm] * n_cols
            table = Table(table_data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(_zebra_table_style(n_cols, len(table_data)))
            story.append(table)
            story.append(Spacer(1, 8))

        # Image (prefer PNG file)
        fig_path = block.get("fig_path")
        if fig_path and os.path.exists(fig_path):
            try:
                story.append(RLImage(fig_path, width=160*mm, height=90*mm))
                story.append(Spacer(1, 8))
            except Exception as e:
                story.append(Paragraph(f"⚠️ Could not embed chart ({e})", BODY))
        else:
            story.append(Paragraph("⚠️ No chart available for this section.", BODY))

        # Insights
        insights = block.get("insights", [])
        if insights:
            wrapped = [textwrap.shorten(str(x), width=120, placeholder="...") for x in insights]
            bullets = "<br/>".join([f"• {b}" for b in wrapped])
            story.append(Paragraph(bullets, BODY))
            summary_insights.append([block.get("title", f"Section {i}"), " ; ".join(wrapped)])
        story.append(PageBreak())

    # === Summary Page ===
    story.append(Paragraph("Executive Summary", H2))
    summary_rows = [["Section", "Key Insights"]] + (summary_insights or [["—", "No insights recorded."]])
    summary_table = Table(summary_rows, colWidths=[60*mm, 100*mm])
    summary_table.setStyle(_zebra_table_style(2, len(summary_rows)))
    story.append(summary_table)

    # Footer Page Numbers
    def _add_page_number(canvas, doc):
        page = canvas.getPageNumber()
        canvas.setFont(DEFAULT_FONT_NAME, 9)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawRightString(A4[0] - 18*mm, 10*mm, f"Page {page}")

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes