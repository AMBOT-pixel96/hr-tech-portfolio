# utils/pdf_auto_exporter.py
"""
PDF Auto Exporter (v2.1)
- export_module_report(...) -> returns PDF bytes
- Accepts data_blocks where each block is a dict:
    {
      "title": str,
      "desc": str,
      "df": pandas.DataFrame (optional),
      "fig": plotly.graph_objects.Figure (optional),
      "insights": [str, str, ...] (optional)
    }
- Uses ReportLab for robust PDF generation and kaleido (plotly) for fig -> PNG.
- For currencies and symbols include a TTF font at utils/fonts/DejaVuSans.ttf (recommended).
"""
# ============================================
# Font Setup — ensures ₹, symbols, and emojis work everywhere
# ============================================

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# Define font path inside Streamlit Cloud (safe & portable)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

try:
    if os.path.exists(FONT_PATH):
        pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_PATH))
    else:
        # fallback — create folder and download font if missing
        os.makedirs("/tmp/fonts", exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
            "/tmp/fonts/DejaVuSans.ttf"
        )
        pdfmetrics.registerFont(TTFont("DejaVuSans", "/tmp/fonts/DejaVuSans.ttf"))
except Exception as e:
    print(f"⚠️ Font setup skipped: {e}")

from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
import os
import pandas as pd
import datetime
import textwrap

# Attempt to register and use DejaVuSans if available for ₹ and wider glyph support
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

DEFAULT_FONT_NAME = "Helvetica"
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
if os.path.exists(FONT_PATH):
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_PATH))
        DEFAULT_FONT_NAME = "DejaVuSans"
    except Exception:
        DEFAULT_FONT_NAME = "Helvetica"

# Helper: convert plotly fig to PNG bytes using kaleido
def fig_to_png_bytes(fig, width=1000, height=600, scale=1):
    """
    Convert a plotly figure to PNG bytes using fig.to_image (kaleido).
    Returns bytes or None if conversion fails.
    """
    try:
        # fig.to_image requires kaleido installed in environment
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        return img_bytes
    except Exception:
        # graceful fallback
        return None

def _df_to_table_data(df: pd.DataFrame, max_rows=25):
    """Convert pandas dataframe to a list-of-lists suitable for ReportLab Table.
    Truncate to max_rows but keep header.
    """
    if df is None or df.empty:
        return [["No data available."]]
    # Convert values to strings and limit column width
    df2 = df.head(max_rows).copy()
    # Convert complex types
    df2 = df2.fillna("").astype(str)
    header = list(df2.columns)
    rows = df2.values.tolist()
    return [header] + rows

def _zebra_table_style(n_cols, n_rows):
    """Return a TableStyle with zebra rows and header styling."""
    style = TableStyle()
    # Header
    style.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A"))
    style.add("TEXTCOLOR", (0, 0), (-1, 0), colors.white)
    style.add("FONTNAME", (0, 0), (-1, 0), DEFAULT_FONT_NAME)
    style.add("FONTSIZE", (0, 0), (-1, -1), 9)
    style.add("ALIGN", (0, 0), (-1, -1), "LEFT")
    style.add("BOTTOMPADDING", (0, 0), (-1, 0), 6)
    # Rows zebra
    for r in range(1, n_rows):
        bg = colors.HexColor("#F7F7F7") if r % 2 == 1 else colors.white
        style.add("BACKGROUND", (0, r), (-1, r), bg)
    style.add("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB"))
    return style

def export_module_report(report_title: str, module_name: str, data_blocks: list, filename_prefix: str = None) -> bytes:
    """
    Compose and return a PDF (bytes) containing:
    - Cover page
    - TOC (zebra table)
    - Each data block section (figure/table + insights)
    - Final summary table of insights
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=15*mm, leftMargin=15*mm,
                            topMargin=18*mm, bottomMargin=18*mm)

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        name="Body", parent=styles["Normal"],
        fontName=DEFAULT_FONT_NAME, fontSize=10, leading=13, textColor=colors.black
    )
    heading_style = ParagraphStyle(
        name="Heading", parent=styles["Heading1"],
        fontName=DEFAULT_FONT_NAME, fontSize=16, leading=20, textColor=colors.HexColor("#0B1220")
    )
    small_muted = ParagraphStyle(
        name="SmallMuted", parent=styles["Normal"],
        fontName=DEFAULT_FONT_NAME, fontSize=9, textColor=colors.HexColor("#374151")
    )

    story = []

    # === Cover Page ===
    story.append(Spacer(1, 20))
    story.append(Paragraph(report_title, heading_style))
    story.append(Spacer(1, 6))
    subtitle_txt = f"{module_name} — Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    story.append(Paragraph(subtitle_txt, small_muted))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Executive Summary", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("This report contains per-metric tables, charts, and concise insights to support board-level review.", body_style))
    story.append(PageBreak())

    # === Table of Contents (zebra) ===
    toc_rows = [["#","Section","Description","Page"]]
    for idx, block in enumerate(data_blocks, start=1):
        desc = block.get("desc","")
        toc_rows.append([str(idx), block.get("title", f"Section {idx}"), (desc if len(desc)<=70 else desc[:67]+"..."), ""])
    toc_table = Table(toc_rows, colWidths=[18*mm, 60*mm, 80*mm, 18*mm])
    n_rows = len(toc_rows)
    toc_table.setStyle(_zebra_table_style(n_cols=4, n_rows=n_rows))
    story.append(Paragraph("Table of Contents", heading_style))
    story.append(Spacer(1,6))
    story.append(toc_table)
    story.append(PageBreak())

    # === Sections ===
    page_index = 2  # cover=1, toc=2
    summary_insights = []
    for idx, block in enumerate(data_blocks, start=1):
        page_index += 1
        title = block.get("title", f"Section {idx}")
        desc = block.get("desc","")
        df = block.get("df", None)
        fig = block.get("fig", None)
        insights = block.get("insights", [])

        # Section header
        story.append(Paragraph(f"{idx}. {title}", ParagraphStyle(
            name="secthead", parent=styles["Heading2"], fontName=DEFAULT_FONT_NAME, fontSize=12)))
        story.append(Paragraph(desc, small_muted))
        story.append(Spacer(1,8))

        # Insert figure if available
        if fig is not None:
            png = fig_to_png_bytes(fig)
            if png:
                img_buf = BytesIO(png)
                rlimg = RLImage(img_buf, width=170*mm, height=(170*mm*0.56))  # maintain aspect-ish ratio
                story.append(rlimg)
                story.append(Spacer(1,6))
            else:
                story.append(Paragraph("⚠️ Chart image could not be rendered. (kaleido missing?)", body_style))
                story.append(Spacer(1,6))

        # Insert dataframe if available
        if df is not None:
            table_data = _df_to_table_data(df, max_rows=30)
            table = Table(table_data, repeatRows=1, hAlign="LEFT")
            table.setStyle(_zebra_table_style(n_cols=len(table_data[0]), n_rows=len(table_data)))
            story.append(table)
            story.append(Spacer(1,6))

        # Insights bullet list
        if insights:
            wrapped = [textwrap.shorten(str(x), width=120, placeholder="...") for x in insights]
            bullets = "<br/>".join([f"• {b}" for b in wrapped])
            story.append(Paragraph(bullets, body_style))
            story.append(Spacer(1,6))
            summary_insights.append({"section": title, "insights": wrapped})
        else:
            story.append(Spacer(1,6))

        story.append(PageBreak())

    # === Final Summary Table ===
    story.append(Paragraph("Consolidated Insights", heading_style))
    summary_rows = [["Section","Key Insights"]]
    for s in summary_insights:
        summary_rows.append([s["section"], " ; ".join(s["insights"])])
    if len(summary_rows)==1:
        summary_rows.append(["None","No insights available."])

    summary_table = Table(summary_rows, colWidths=[60*mm, 100*mm])
    summary_table.setStyle(_zebra_table_style(n_cols=2, n_rows=len(summary_rows)))
    story.append(summary_table)

    # === Build ===
    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes