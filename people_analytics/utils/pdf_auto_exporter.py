# ============================================
# utils/pdf_auto_exporter.py — v3.6 | Renderer Independence Edition
# ============================================

from io import BytesIO
import os, datetime, textwrap
import pandas as pd
import plotly.io as pio
import plotly.express as px
from PIL import Image
import base64
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, Paragraph as RLParagraph
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------
# 🧩 Font Setup
# ---------------------------
DEFAULT_FONT_NAME = "DejaVuSans"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    pdfmetrics.registerFont(TTFont(DEFAULT_FONT_NAME, FONT_PATH))
except Exception as e:
    print(f"⚠️ Font registration skipped: {e}")
    DEFAULT_FONT_NAME = "Helvetica"

# ---------------------------
# 🧠 Styles
# ---------------------------
styles = getSampleStyleSheet()
BODY_STYLE = ParagraphStyle("Body", parent=styles["Normal"], fontName=DEFAULT_FONT_NAME, fontSize=10, leading=12)
TITLE_STYLE = ParagraphStyle("Title", parent=styles["Title"], fontName=DEFAULT_FONT_NAME, fontSize=20, alignment=1)
SUBTITLE_STYLE = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName=DEFAULT_FONT_NAME, fontSize=11, alignment=1, textColor=colors.HexColor("#374151"))
H2_STYLE = ParagraphStyle("H2", parent=styles["Heading2"], fontName=DEFAULT_FONT_NAME, fontSize=14, textColor=colors.HexColor("#111827"))
TABLE_PAR_STYLE = ParagraphStyle("TableCell", fontName=DEFAULT_FONT_NAME, fontSize=9, leading=11)

# ---------------------------
# 🎨 Plotly Theme Helper
# ---------------------------
DEFAULT_COLORWAY = px.colors.qualitative.Plotly

def apply_bright_theme(fig):
    """Ensure bright, PDF-safe color theme."""
    fig.update_layout(
        template="plotly_white",
        colorway=DEFAULT_COLORWAY,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="black",
        margin=dict(t=60, b=40, l=40, r=40),
        showlegend=True
    )
    for tr in fig.data:
        if hasattr(tr, "marker"):
            tr.marker.line = dict(color="black", width=1)
    return fig

# ---------------------------
# 🖼️ Universal Renderer (No Kaleido)
# ---------------------------
def fig_to_png_bytes(fig, width=900, height=520):
    """
    Convert Plotly figure to PNG bytes without Kaleido.
    Uses to_html + base64 + Pillow fallback.
    """
    try:
        fig = apply_bright_theme(fig)
        # Get HTML and render via base64 export
        img_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=1)
        return img_bytes
    except Exception as e:
        print(f"⚠️ Kaleido fallback invoked due to: {e}")
        try:
            # Secondary route: via HTML screenshot
            html = pio.to_html(fig, full_html=False)
            b64_start = html.find("base64,") + len("base64,")
            b64_end = html.find('"', b64_start)
            b64_data = html[b64_start:b64_end]
            img_data = base64.b64decode(b64_data)
            img = Image.open(BytesIO(img_data))
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e2:
            print(f"🚨 HTML conversion failed: {e2}")
            return None

# ---------------------------
# 🧾 Table Helpers
# ---------------------------
def _format_val(v):
    try:
        if v is None:
            return ""
        if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer()):
            return str(int(v))
        if isinstance(v, float):
            return f"{v:.2f}"
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return f"{f:.2f}"
    except Exception:
        return str(v)

def _cell_val(v):
    text = _format_val(v)
    return RLParagraph(str(text), TABLE_PAR_STYLE)

def _df_to_table_data(df: pd.DataFrame, max_rows=15, max_cols=6):
    """Convert DataFrame → wrapped table, truncated for exec readability."""
    if df is None or df.empty:
        return [[RLParagraph("No data available.", TABLE_PAR_STYLE)]]
    if df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols]
    if df.shape[0] > max_rows:
        df = df.head(max_rows)

    df2 = df.copy()
    for c in df2.select_dtypes(include=["float", "int"]).columns:
        df2[c] = df2[c].round(2)
    for c in df2.columns:
        df2[c] = df2[c].astype(str)
    header = [RLParagraph(str(h), ParagraphStyle("Header", fontName=DEFAULT_FONT_NAME, fontSize=9, textColor=colors.white)) for h in df2.columns]
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

# ---------------------------
# 📄 PDF Exporter
# ---------------------------
def export_module_report(report_title: str, module_name: str, data_blocks: list, filename_prefix: str = None) -> bytes:
    """
    Compose and return a PDF (bytes) containing:
    - Cover page
    - TOC
    - Section tables & charts
    - Summary page
    """
    # Pre-render all figures to PNG bytes first
    for block in data_blocks:
        fig = block.get("fig")
        if fig is not None:
            img_bytes = fig_to_png_bytes(fig)
            block["__img_bytes__"] = img_bytes
        else:
            block["__img_bytes__"] = None

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=22*mm, bottomMargin=18*mm)
    story = []

    # Cover Page
    story.append(Spacer(1, 80))
    story.append(Paragraph(report_title, TITLE_STYLE))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"{module_name} Module", SUBTITLE_STYLE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", SUBTITLE_STYLE))
    story.append(Spacer(1, 80))
    story.append(Paragraph("<b>Prepared by:</b> Amlan Mishra", SUBTITLE_STYLE))
    story.append(PageBreak())

    # Table of Contents
    toc_rows = [["#", "Section", "Description", "Page"]]
    for i, block in enumerate(data_blocks, 1):
        desc = block.get("desc", "")
        if len(desc) > 70:
            desc = desc[:67] + "..."
        toc_rows.append([str(i), block.get("title", f"Section {i}"), desc, str(i + 2)])
    toc_table = Table(toc_rows, colWidths=[15*mm, 65*mm, 85*mm, 15*mm])
    toc_table.setStyle(_zebra_table_style(4, len(toc_rows)))
    story.append(Paragraph("Table of Contents", H2_STYLE))
    story.append(Spacer(1, 6))
    story.append(toc_table)
    story.append(PageBreak())

    # Sections
    summary_insights = []
    for i, block in enumerate(data_blocks, 1):
        title = block.get("title", f"Section {i}")
        desc = block.get("desc", "")
        df = block.get("df", None)
        img_bytes = block.get("__img_bytes__", None)
        insights = block.get("insights", [])

        story.append(Paragraph(f"{i}. {title}", H2_STYLE))
        if desc:
            story.append(Spacer(1, 4))
            story.append(Paragraph(desc, BODY_STYLE))
        story.append(Spacer(1, 8))

        # Table
        if df is not None:
            table_data = _df_to_table_data(df)
            n_cols = len(table_data[0])
            col_widths = [max(30, min(55, 500 / n_cols)) * mm] * n_cols
            table = Table(table_data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(_zebra_table_style(n_cols, len(table_data)))
            story.append(table)
            story.append(Spacer(1, 8))

        # Figure
        if img_bytes:
            try:
                rlimg = RLImage(BytesIO(img_bytes), width=160*mm, height=90*mm)
                story.append(rlimg)
                story.append(Spacer(1, 8))
            except Exception as e:
                story.append(Paragraph(f"⚠️ Graph embed failed: {e}", BODY_STYLE))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("⚠️ Graph rendering unavailable.", BODY_STYLE))
            story.append(Spacer(1, 8))

        # Insights
        if insights:
            wrapped = [textwrap.shorten(str(x), width=120, placeholder="...") for x in insights]
            bullets = "<br/>".join([f"• {b}" for b in wrapped])
            story.append(Paragraph(bullets, BODY_STYLE))
            summary_insights.append([title, " ; ".join(wrapped)])
        story.append(PageBreak())

    # Summary Page
    story.append(Paragraph("Executive Summary", H2_STYLE))
    if not summary_insights:
        summary_insights = [["—", "No insights recorded."]]
    summary_table_rows = [["Section", "Key Insights"]] + summary_insights
    summary_table = Table(summary_table_rows, colWidths=[60*mm, 100*mm])
    summary_table.setStyle(_zebra_table_style(2, len(summary_table_rows)))
    story.append(summary_table)

    # Footer Page Numbers
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