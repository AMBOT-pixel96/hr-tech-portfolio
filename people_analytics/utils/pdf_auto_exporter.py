# ============================================
# utils/pdf_auto_exporter.py — v3.5-Stable | Synchronous Stream Edition
# ============================================

from io import BytesIO
import os, datetime, textwrap
from time import sleep
import pandas as pd
import plotly.express as px
import plotly.io as pio
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
# Kaleido init (best-effort)
# ---------------------------
try:
    pio.renderers.default = "kaleido"
except Exception as e:
    print(f"⚠️ Kaleido init failed: {e}")

# ---------------------------
# Font setup
# ---------------------------
DEFAULT_FONT_NAME = "DejaVuSans"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    pdfmetrics.registerFont(TTFont(DEFAULT_FONT_NAME, FONT_PATH))
except Exception as e:
    print(f"⚠️ Font registration skipped: {e}")
    DEFAULT_FONT_NAME = "Helvetica"

# ---------------------------
# Styles
# ---------------------------
styles = getSampleStyleSheet()
BODY_STYLE = ParagraphStyle("Body", parent=styles["Normal"], fontName=DEFAULT_FONT_NAME, fontSize=10, leading=12)
TITLE_STYLE = ParagraphStyle("Title", parent=styles["Title"], fontName=DEFAULT_FONT_NAME, fontSize=20, alignment=1)
SUBTITLE_STYLE = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName=DEFAULT_FONT_NAME, fontSize=11, alignment=1, textColor=colors.HexColor("#374151"))
H2_STYLE = ParagraphStyle("H2", parent=styles["Heading2"], fontName=DEFAULT_FONT_NAME, fontSize=14, textColor=colors.HexColor("#111827"))
TABLE_PAR_STYLE = ParagraphStyle("TableCell", fontName=DEFAULT_FONT_NAME, fontSize=9, leading=11)

# ---------------------------
# Color & theme helpers
# ---------------------------
DEFAULT_COLORWAY = px.colors.qualitative.Plotly

def apply_bright_theme(fig):
    """Apply a PDF-safe bright theme and keep a colorway."""
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="black",
        colorway=DEFAULT_COLORWAY,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    # add subtle outlines where possible
    for tr in fig.data:
        if hasattr(tr, "marker"):
            tr.marker.line = dict(color="black", width=1)
    return fig

# ---------------------------
# Figure -> PNG bytes (stable)
# ---------------------------
def fig_to_png_bytes(fig, width=900, height=520, scale=1):
    """
    Convert Plotly fig -> PNG bytes with:
      - explicit color bake (no deepcopy — preserve cached state)
      - 2-attempt retry (smaller dims on 2nd attempt)
      - opaque white background (RGBA)
    Returns bytes or None.
    """
    try:
        # apply theme (do not deepcopy to avoid expensive cache invalidation)
        fig = apply_bright_theme(fig)

        # bake explicit trace colors if missing
        palette = px.colors.qualitative.Plotly
        for i, tr in enumerate(fig.data):
            base_color = palette[i % len(palette)]
            if hasattr(tr, "marker"):
                if not getattr(tr.marker, "color", None):
                    tr.marker.color = base_color
                tr.marker.line = dict(color="black", width=1)
            if hasattr(tr, "line"):
                if not getattr(tr.line, "color", None):
                    tr.line.color = base_color
                if not getattr(tr.line, "width", None):
                    tr.line.width = 2

        # ensure layout explicitness
        fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white", font_color="black")

        attempts = [
            (width, height, scale),
            (max(600, int(width * 0.75)), max(360, int(height * 0.75)), 1)
        ]

        last_err = None
        for idx, (w, h, s) in enumerate(attempts, start=1):
            try:
                img = fig.to_image(
                    format="png",
                    engine="kaleido",
                    width=w,
                    height=h,
                    scale=s,
                    validate=True,
                    background='rgba(255,255,255,1.0)'
                )
                print(f"🎨 fig exported (attempt {idx}) size={w}x{h}")
                return img
            except Exception as e:
                last_err = e
                print(f"⚠️ Kaleido attempt {idx} failed: {e}")
                sleep(0.7)

        print(f"❌ fig export failed after retries: {last_err}")
        return None

    except Exception as e:
        print(f"🚨 fig_to_png_bytes fatal: {e}")
        return None

# ---------------------------
# Table helpers
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
    """Convert DataFrame to wrapped table; truncate to avoid raw dumps."""
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
# PDF exporter
# ---------------------------
def export_module_report(report_title: str, module_name: str, data_blocks: list, filename_prefix: str = None) -> bytes:
    """
    Build PDF bytes. IMPORTANT: pre-renders all figures (synchronously) into PNG bytes
    so PDF building only starts after all images are ready.
    """
    # Pre-render figures first (synchronous)
    for block in data_blocks:
        fig = block.get("fig", None)
        block["__img_bytes__"] = None
        if fig is not None:
            img_bytes = fig_to_png_bytes(fig)
            if img_bytes is None:
                # mark failure; exporter will render a warning text
                block["__render_failed__"] = True
                block["__img_bytes__"] = None
            else:
                block["__render_failed__"] = False
                block["__img_bytes__"] = img_bytes
        else:
            block["__render_failed__"] = False
            block["__img_bytes__"] = None

    # Now build PDF using the pre-rendered bytes
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=22*mm, bottomMargin=18*mm)
    story = []

    # Cover
    story.append(Spacer(1, 80))
    story.append(Paragraph(report_title, TITLE_STYLE))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"{module_name} Module", SUBTITLE_STYLE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", SUBTITLE_STYLE))
    story.append(Spacer(1, 80))
    story.append(Paragraph("<b>Prepared by:</b> Amlan Mishra", SUBTITLE_STYLE))
    story.append(PageBreak())

    # TOC (wider columns to avoid spill)
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
        render_failed = block.get("__render_failed__", False)
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

        # Figure (use pre-rendered bytes)
        if img_bytes:
            try:
                rlimg = RLImage(BytesIO(img_bytes), width=160*mm, height=90*mm)
                story.append(rlimg)
                story.append(Spacer(1, 8))
            except Exception as e:
                story.append(Paragraph("⚠️ Graph could not be embedded.", BODY_STYLE))
                story.append(Spacer(1, 8))
        else:
            if render_failed:
                story.append(Paragraph("⚠️ Graph rendering failed (Kaleido).", BODY_STYLE))
                story.append(Spacer(1, 8))

        # Insights
        if insights:
            wrapped = [textwrap.shorten(str(x), width=120, placeholder="...") for x in insights]
            bullets = "<br/>".join([f"• {b}" for b in wrapped])
            story.append(Paragraph(bullets, BODY_STYLE))
            summary_insights.append([title, " ; ".join(wrapped)])
        story.append(PageBreak())

    # Summary
    story.append(Paragraph("Executive Summary", H2_STYLE))
    if not summary_insights:
        summary_insights = [["—", "No insights recorded."]]
    summary_table_rows = [["Section", "Key Insights"]] + summary_insights
    summary_table = Table(summary_table_rows, colWidths=[60*mm, 100*mm])
    summary_table.setStyle(_zebra_table_style(2, len(summary_table_rows)))
    story.append(summary_table)

    # Footer page numbers
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