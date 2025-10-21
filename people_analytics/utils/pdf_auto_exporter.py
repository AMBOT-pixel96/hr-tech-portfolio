# ============================================
# utils/pdf_auto_exporter.py — v2.2 | Improved Kaleido & TOC Layout
# ============================================

from io import BytesIO
import os, datetime, textwrap
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------
# 🔤 Font Setup
# ---------------------------
DEFAULT_FONT_NAME = "Helvetica"
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
try:
    if os.path.exists(FONT_PATH):
        pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_PATH))
        DEFAULT_FONT_NAME = "DejaVuSans"
except Exception as e:
    print(f"⚠️ Font registration skipped: {e}")

# ---------------------------
# 🖼 Plotly Figure to PNG
# ---------------------------
def fig_to_png_bytes(fig, width=1000, height=600, scale=1):
    """Convert a Plotly figure to PNG bytes with safe Kaleido fallback."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception as e:
        print(f"⚠️ Kaleido export failed: {e}")
        return None

# ---------------------------
# 📊 DataFrame → ReportLab Table
# ---------------------------
def _df_to_table_data(df: pd.DataFrame, max_rows=25):
    if df is None or df.empty:
        return [["No data available."]]
    df2 = df.head(max_rows).copy().fillna("").astype(str)
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
        bg = colors.HexColor("#F7F7F7") if r % 2 == 1 else colors.white
        style.add("BACKGROUND", (0, r), (-1, r), bg)
    style.add("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB"))
    style.add("ALIGN", (0, 0), (-1, -1), "LEFT")
    style.add("BOTTOMPADDING", (0, 0), (-1, 0), 6)
    return style

# ---------------------------
# 🧾 Export Function
# ---------------------------
def export_module_report(report_title: str, module_name: str, data_blocks: list, filename_prefix: str = None) -> bytes:
    """
    Compose and return a PDF report (bytes) with:
      • Cover page
      • Table of Contents
      • Section data (chart + table + insights)
      • Consolidated summary page
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=15*mm, leftMargin=15*mm,
                            topMargin=18*mm, bottomMargin=18*mm)

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                fontName=DEFAULT_FONT_NAME, fontSize=10,
                                leading=13, textColor=colors.black)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading1"],
                                   fontName=DEFAULT_FONT_NAME, fontSize=16,
                                   leading=20, textColor=colors.HexColor("#0B1220"))
    small_muted = ParagraphStyle("SmallMuted", parent=styles["Normal"],
                                 fontName=DEFAULT_FONT_NAME, fontSize=9,
                                 textColor=colors.HexColor("#374151"))

    story = []

    # ---------------------------
    # 🧠 Cover Page
    # ---------------------------
    story.append(Spacer(1, 20))
    story.append(Paragraph(report_title, heading_style))
    story.append(Spacer(1, 6))
    subtitle = f"{module_name} — Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    story.append(Paragraph(subtitle, small_muted))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Executive Summary", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("This report contains per-metric tables, charts, and concise insights for leadership review.", body_style))
    story.append(PageBreak())

    # ---------------------------
    # 📖 Table of Contents
    # ---------------------------
    toc_rows = [["#", "Section", "Description", "Page"]]
    for idx, block in enumerate(data_blocks, start=1):
        desc = block.get("desc", "")
        desc = desc[:67] + "..." if len(desc) > 70 else desc
        toc_rows.append([str(idx), block.get("title", f"Section {idx}"), desc, "—"])
    toc_table = Table(toc_rows, colWidths=[18*mm, 60*mm, 80*mm, 18*mm])
    toc_table.setStyle(_zebra_table_style(4, len(toc_rows)))
    story.append(Paragraph("Table of Contents", heading_style))
    story.append(Spacer(1, 6))
    story.append(toc_table)
    story.append(PageBreak())

    # ---------------------------
    # 📊 Sections
    # ---------------------------
    summary_insights = []
    for idx, block in enumerate(data_blocks, start=1):
        title = block.get("title", f"Section {idx}")
        desc = block.get("desc", "")
        df = block.get("df", None)
        fig = block.get("fig", None)
        insights = block.get("insights", [])

        story.append(Paragraph(f"{idx}. {title}", ParagraphStyle("secthead",
                            parent=styles["Heading2"], fontName=DEFAULT_FONT_NAME,
                            fontSize=12)))
        story.append(Paragraph(desc, small_muted))
        story.append(Spacer(1, 8))

        # Chart (if provided)
        if fig is not None:
            img_bytes = fig_to_png_bytes(fig)
            if img_bytes:
                rl_img = RLImage(BytesIO(img_bytes), width=170*mm, height=(170*mm*0.56))
                story.append(rl_img)
                story.append(Spacer(1, 6))
            else:
                story.append(Paragraph("⚠️ Chart could not be rendered (Kaleido missing).", body_style))

        # Data table
        if df is not None:
            table_data = _df_to_table_data(df, max_rows=25)
            table = Table(table_data, repeatRows=1, hAlign="LEFT")
            table.setStyle(_zebra_table_style(len(table_data[0]), len(table_data)))
            story.append(table)
            story.append(Spacer(1, 6))

        # Insights
        if insights:
            wrapped = [textwrap.shorten(str(x), width=120, placeholder="...") for x in insights]
            bullets = "<br/>".join([f"• {b}" for b in wrapped])
            story.append(Paragraph(bullets, body_style))
            summary_insights.append({"section": title, "insights": wrapped})
        story.append(PageBreak())

    # ---------------------------
    # 🧩 Consolidated Insights
    # ---------------------------
    story.append(Paragraph("Consolidated Insights", heading_style))
    summary_rows = [["Section", "Key Insights"]]
    for s in summary_insights:
        summary_rows.append([s["section"], " ; ".join(s["insights"])])
    if len(summary_rows) == 1:
        summary_rows.append(["None", "No insights recorded."])

    summary_table = Table(summary_rows, colWidths=[60*mm, 100*mm])
    summary_table.setStyle(_zebra_table_style(2, len(summary_rows)))
    story.append(summary_table)

    # Build & return PDF
    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes