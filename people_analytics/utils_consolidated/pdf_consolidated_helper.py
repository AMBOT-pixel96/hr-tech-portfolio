# utils_consolidated/pdf_consolidated_helper.py
"""
Consolidated PDF builder. Exposes `render_consolidated_pdf(report_title, module_label, data_blocks, file_prefix)`
data_blocks is a list of dicts like:
  {"title": "...", "desc":"...", "df": pd.DataFrame or None, "fig": plotly fig or None, "insights": [...]}
This file uses utilities from utils_consolidated.
"""

import os
import io
import time
import streamlit as st
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# local consolidated utilities
from .constants import PAGE_SIZE, MARGINS, COVER, DEFAULT_FONT, CHART_WIDTH_MM, CHART_HEIGHT_MM
from .chart_consolidated_saver import ensure_chart_saved
from .toc_helper import build_toc_entries, toc_colwidths
from .insights_helper import flatten_insights

# Ensure DejaVu font registration for unicode (₹ etc.)
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = DEFAULT_FONT  # fallback

def _pstyle(name="body", size=10, bold=False, color=colors.black):
    styles = getSampleStyleSheet()
    base = styles["Normal"]
    return ParagraphStyle(
        name,
        parent=base,
        fontName="DejaVuSans" if DEFAULT_FONT == "DejaVuSans" else "Helvetica",
        fontSize=size,
        leading=int(size * 1.2),
        textColor=color,
    )

def render_consolidated_pdf(report_title, module_label, data_blocks, file_prefix):
    if not data_blocks:
        st.warning("⚠️ No data blocks to export.")
        return

    if not st.button(f"🧾 Generate {module_label} Executive PDF", use_container_width=True):
        return

    buf = io.BytesIO()
    try:
        doc = SimpleDocTemplate(
            buf,
            pagesize=PAGE_SIZE,
            rightMargin=MARGINS["right"],
            leftMargin=MARGINS["left"],
            topMargin=MARGINS["top"],
            bottomMargin=MARGINS["bottom"],
        )

        story = []

        # COVER
        story.append(Spacer(1, 40))
        story.append(Paragraph(f"<para align=center><font size={COVER['title_size']}><b>{report_title}</b></font></para>", _pstyle("cover", COVER['title_size'], bold=True, color=COVER['title_color'])))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<para align=center><font size={COVER['subtitle_size']}>{module_label}</font></para>", _pstyle("subtitle", COVER['subtitle_size'], color=COVER['subtitle_color'])))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<para align=center><font size=9>Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}</font></para>", _pstyle("meta", 9)))
        story.append(PageBreak())

        # TOC
        toc_rows = build_toc_entries(data_blocks)
        toc_table = Table(toc_rows, colWidths=toc_colwidths())
        toc_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E5E7EB")),
            ("GRID", (0,0), (-1,-1), 0.25, colors.black),
            ("FONTNAME", (0,0), (-1,0), "DejaVuSans"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(Paragraph("<b>Table of Contents</b>", _pstyle("h2", 12, bold=True)))
        story.append(Spacer(1,6))
        story.append(toc_table)
        story.append(PageBreak())

        # Sections
        summary_rows = [["Section", "Key Insights"]]
        for idx, block in enumerate(data_blocks, start=1):
            title = block.get("title", f"Section {idx}")
            desc = block.get("desc", "")
            df = block.get("df", None)
            fig = block.get("fig", None)
            insights = block.get("insights", [])

            story.append(Paragraph(f"{idx}. {title}", _pstyle("section_title", 12, bold=True, color=colors.HexColor("#0F172A"))))
            if desc:
                story.append(Paragraph(desc, _pstyle("desc", 9, color=colors.HexColor("#374151"))))
            story.append(Spacer(1,6))

            # Data table
            if df is not None and hasattr(df, "empty") and not df.empty:
                try:
                    df2 = df.copy().round(2).astype(str)
                    table_data = [list(df2.columns)] + df2.values.tolist()
                    col_count = len(df2.columns)
                    colw = (PAGE_SIZE[0] - (MARGINS["left"] + MARGINS["right"])) / max(1, col_count)
                    table = Table(table_data, colWidths=[colw]*col_count, repeatRows=1)
                    table.setStyle(TableStyle([
                        ("GRID", (0,0), (-1,-1), 0.25, colors.black),
                        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F3F4F6")),
                        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
                        ("FONTSIZE", (0,0), (-1,-1), 9),
                        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                    ]))
                    story.append(table)
                    story.append(Spacer(1,8))
                except Exception as e:
                    story.append(Paragraph(f"⚠️ Table render error: {e}", _pstyle("err", 8, color=colors.red)))
                    story.append(Spacer(1,6))

            # Chart export (plotly fig -> PNG)
            if fig is not None:
                img_path = None
                try:
                    # ensure_chart_saved will retry and return a filesystem path or None
                    img_path = ensure_chart_saved(title, fig)
                    # wait a tad
                    for _ in range(6):
                        if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                            break
                        time.sleep(0.15)
                    if img_path and os.path.exists(img_path):
                        story.append(RLImage(img_path, width=CHART_WIDTH_MM * mm, height=CHART_HEIGHT_MM * mm))
                        story.append(Spacer(1,8))
                    else:
                        story.append(Paragraph("⚠️ Chart could not be rendered.", _pstyle("err", 9, color=colors.HexColor("#B91C1C"))))
                        story.append(Spacer(1,6))
                except Exception as e:
                    story.append(Paragraph(f"⚠️ Chart render error: {e}", _pstyle("err", 9, color=colors.HexColor("#B91C1C"))))
                    story.append(Spacer(1,6))
            else:
                story.append(Paragraph("⚠️ No chart available for this section.", _pstyle("note", 9, color=colors.HexColor("#6B7280"))))
                story.append(Spacer(1,6))

            # Insights
            combined = flatten_insights(insights)
            if combined:
                story.append(Paragraph(f"<i>{combined}</i>", _pstyle("ins", 9, color=colors.HexColor("#2563EB"))))
                story.append(Spacer(1,8))

            summary_rows.append([title, combined])
            story.append(PageBreak())

        # Executive summary page
        story.append(Paragraph("Executive Summary", _pstyle("summary_h", 13, bold=True)))
        story.append(Spacer(1, 8))
        try:
            summary_table = Table(summary_rows, colWidths=[140, 310])
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E5E7EB")),
                ("GRID", (0,0), (-1,-1), 0.25, colors.black),
                ("FONTNAME", (0,0), (-1,0), "DejaVuSans"),
                ("FONTSIZE", (0,0), (-1,-1), 9),
            ]))
            story.append(summary_table)
            story.append(Spacer(1,8))
        except Exception as e:
            story.append(Paragraph(f"⚠️ Executive summary render error: {e}", _pstyle("err", 9, color=colors.red)))
            story.append(Spacer(1,6))

        # Build PDF
        doc.build(story)
        pdf_bytes = buf.getvalue()
        st.success("✅ Consolidated Executive PDF generated.")
        st.download_button(
            "⬇️ Download Consolidated Report",
            pdf_bytes,
            file_name=f"{file_prefix}_Consolidated_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"⚠️ PDF build failed: {e}")
        st.write(traceback.format_exc())
    finally:
        buf.close()