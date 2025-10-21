# utils/pdf_auto_exporter.py — v3.1 | Executive PDF Engine (wrapping, 2-decimal, colored charts)
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

# Kaleido init (ensure renderer)
try:
    pio.renderers.default = "kaleido"
except Exception as e:
    print(f"⚠️ Kaleido init failed: {e}")

# Font registration (DejaVu for wider glyph support)
DEFAULT_FONT_NAME = "DejaVuSans"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    pdfmetrics.registerFont(TTFont(DEFAULT_FONT_NAME, FONT_PATH))
except Exception as e:
    print(f"⚠️ Font registration skipped (fallback to default): {e}")
    DEFAULT_FONT_NAME = "Helvetica"

styles = getSampleStyleSheet()
BODY_STYLE = ParagraphStyle("Body", parent=styles["Normal"], fontName=DEFAULT_FONT_NAME, fontSize=10, leading=12)
TITLE_STYLE = ParagraphStyle("Title", parent=styles["Title"], fontName=DEFAULT_FONT_NAME, fontSize=20, alignment=1)
SUBTITLE_STYLE = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName=DEFAULT_FONT_NAME, fontSize=11, alignment=1, textColor=colors.HexColor("#374151"))
H2_STYLE = ParagraphStyle("H2", parent=styles["Heading2"], fontName=DEFAULT_FONT_NAME, fontSize=14, textColor=colors.HexColor("#111827"))

# helper: format numbers to max 2 decimals
def _format_val(v):
    try:
        if v is None:
            return ""
        if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer()):
            return str(int(v))
        if isinstance(v, float):
            return f"{v:.2f}"
        # try numeric string
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return f"{f:.2f}"
    except Exception:
        return str(v)

# helper: wrap text into Paragraph for table cells
from reportlab.platypus import Paragraph as RLParagraph
from reportlab.lib.styles import ParagraphStyle
TABLE_PAR_STYLE = ParagraphStyle("TableCell", fontName=DEFAULT_FONT_NAME, fontSize=9, leading=11)

def _cell_val(v):
    """Return a Paragraph-wrapped and formatted value for table cell"""
    # format numeric to 2 decimals
    text = _format_val(v)
    # safe-escape
    return RLParagraph(str(text), TABLE_PAR_STYLE)

# convert dataframe to table data (wrapped Paragraphs)
def _df_to_table_data(df: pd.DataFrame, max_rows=20):
    if df is None or df.empty:
        return [[RLParagraph("No data available.", TABLE_PAR_STYLE)]]
    df2 = df.head(max_rows).copy()
    # Format numeric columns
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].apply(lambda x: _format_val(x))
        else:
            df2[c] = df2[c].fillna("").astype(str)
    header = [RLParagraph(str(h), ParagraphStyle("h", fontName=DEFAULT_FONT_NAME, fontSize=9, textColor=colors.white)) for h in df2.columns]
    rows = []
    for _, r in df2.iterrows():
        rows.append([_cell_val(v) for v in r.tolist()])
    return [header] + rows

def _zebra_table_style(n_cols, n_rows):
    style = TableStyle()
    style.add("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT_NAME)
    style.add("FONTSIZE", (0, 0), (-1, -1), 9)
    style.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A"))
    style.add("TEXTCOLOR", (0, 0), (-1, 0), colors.white)
    for r in range(1, n_rows):
        bg = colors.HexColor("#F7F7F7") if r % 2 == 1 else colors.white
        style.add("BACKGROUND", (0, r), (-1, r), bg)
    style.add("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB"))
    style.add("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    return style

# convert plotly fig to png bytes (force bright template for PDF)
def fig_to_png_bytes(fig, width=900, height=520, scale=1):
    try:
        # force white template for PDF export (keeps color palettes)
        fig.update_layout(template="plotly_white",
                          plot_bgcolor="white", paper_bgcolor="white",
                          font_color="black")
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        return img_bytes
    except Exception as e:
        print(f"⚠️ Kaleido export failed: {e}")
        return None

# The exporter: builds cover, toc, each section (one page), summary, returns bytes
def export_module_report(report_title: str, module_name: str, data_blocks: list, filename_prefix: str = None) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=22*mm, bottomMargin=18*mm)
    story = []

    # Cover
    story.append(Spacer(1, 80))
    story.append(Paragraph(report_title, TITLE_STYLE))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"{module_name} Module", SUBTITLE_STYLE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", SUBTITLE_STYLE))
    story.append(Spacer(1, 80))
    story.append(Paragraph(f"<b>Prepared by:</b> Amlan Mishra", SUBTITLE_STYLE))
    story.append(PageBreak())

    # TOC (page numbers: cover=1, toc=2, section i start page = i+2 because each section is 1 page)
    toc_rows = [["#", "Section", "Description", "Page"]]
    for i, block in enumerate(data_blocks, 1):
        desc = block.get("desc", "")
        if len(desc) > 70: desc = desc[:67] + "..."
        toc_rows.append([str(i), block.get("title", f"Section {i}"), desc, str(i + 2)])
    toc_table = Table(toc_rows, colWidths=[15*mm, 60*mm, 80*mm, 15*mm])
    toc_table.setStyle(_zebra_table_style(4, len(toc_rows)))
    story.append(Paragraph("Table of Contents", H2_STYLE))
    story.append(Spacer(1, 6))
    story.append(toc_table)
    story.append(PageBreak())

    # Sections: one page per data_block (Metric -> Table -> Graph -> Insight)
    summary_insights = []
    for i, block in enumerate(data_blocks, 1):
        title = block.get("title", f"Section {i}")
        desc = block.get("desc", "")
        df = block.get("df", None)
        fig = block.get("fig", None)
        insights = block.get("insights", [])

        story.append(Paragraph(f"{i}. {title}", H2_STYLE))
        if desc:
            story.append(Spacer(1, 4))
            story.append(Paragraph(desc, BODY_STYLE))
        story.append(Spacer(1, 8))

        # Table (summary - limited rows)
        if df is not None:
            table_data = _df_to_table_data(df, max_rows=25)
            table = Table(table_data, repeatRows=1, hAlign="LEFT")
            table.setStyle(_zebra_table_style(len(table_data[0]), len(table_data)))
            story.append(table)
            story.append(Spacer(1, 8))

        # Graph
        if fig is not None:
            img_bytes = fig_to_png_bytes(fig)
            if img_bytes:
                rlimg = RLImage(BytesIO(img_bytes), width=160*mm, height=90*mm)
                story.append(rlimg)
                story.append(Spacer(1, 8))
            else:
                story.append(Paragraph("⚠️ Graph rendering failed (Kaleido).", BODY_STYLE))
                story.append(Spacer(1, 8))

        # Insights
        if insights:
            wrapped = [textwrap.shorten(str(x), width=120, placeholder="...") for x in insights]
            bullets = "<br/>".join([f"• {b}" for b in wrapped])
            story.append(Paragraph(bullets, BODY_STYLE))
            summary_insights.append([title, " ; ".join(wrapped)])

        # Ensure one page per section
        story.append(PageBreak())

    # Summary page
    story.append(Paragraph("Executive Summary", H2_STYLE))
    if not summary_insights:
        summary_insights = [["—", "No insights recorded."]]
    summary_table_rows = [["Section", "Key Insights"]] + summary_insights
    summary_table = Table(summary_table_rows, colWidths=[60*mm, 100*mm])
    summary_table.setStyle(_zebra_table_style(2, len(summary_table_rows)))
    story.append(summary_table)

    # Footer & page numbers via canvas callback (draw on build)
    def _add_page_number(canvas, doc):
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.setFont(DEFAULT_FONT_NAME, 9)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawRightString(A4[0] - 18*mm, 10*mm, text)

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes