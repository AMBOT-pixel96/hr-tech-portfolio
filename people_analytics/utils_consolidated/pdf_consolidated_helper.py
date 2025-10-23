# utils_consolidated/pdf_consolidated_helper.py — v5.1 Boardroom Edition
import os
import io
import time
import traceback
from datetime import datetime

import streamlit as st
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# try to import helpers from your utils (adjust import if your module path differs)
try:
    from utils.chart_saver import ensure_chart_saved, save_chart_image
except Exception:
    # fallback names — if you put consolidated utils in a different package, adjust import
    try:
        from utils_consolidated.chart_consolidated_saver import ensure_chart_saved, save_chart_image
    except Exception:
        ensure_chart_saved = None
        save_chart_image = None

# -------------------------------------------------------
# Fonts: register DejaVuSans for currency / unicode support
# -------------------------------------------------------
DEFAULT_FONT = "Helvetica"
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    # keep DEFAULT_FONT as Helvetica if DejaVu not available
    pass

# -------------------------------------------------------
# Constants for layout (Boardroom Edition v5.1)
# -------------------------------------------------------
PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = RIGHT_MARGIN = 18 * mm
TOP_MARGIN = BOTTOM_MARGIN = 20 * mm

CHART_W_MM = 170
CHART_H_MM = 95

TOC_COLWIDTHS = [25, 110, 240, 35]

# color palette used in document
PRIMARY_COLOR = colors.HexColor("#111827")
ACCENT_COLOR = colors.HexColor("#2563EB")
SECTION_COLOR = colors.HexColor("#1E3A8A")
TOC_HEADER_BG = colors.HexColor("#E5E7EB")
TABLE_HEADER_BG = colors.HexColor("#F3F4F6")
HIGHLIGHT = colors.HexColor("#FACC15")

# -------------------------------------------------------
# Helpers for the PDF builder
# -------------------------------------------------------
def _para_style(name="body", **kwargs):
    styles = getSampleStyleSheet()
    base = styles.get("Normal")
    return ParagraphStyle(name, parent=base, **kwargs)

BODY_STYLE = _para_style("body", fontName=DEFAULT_FONT, fontSize=10, leading=13, textColor=PRIMARY_COLOR)
HEADING_STYLE = _para_style("heading", fontName=DEFAULT_FONT, fontSize=13, leading=16, textColor=SECTION_COLOR)
COVER_TITLE = _para_style("cover_title", fontName=DEFAULT_FONT, fontSize=22, leading=26, textColor=PRIMARY_COLOR, alignment=1)
COVER_SUB = _para_style("cover_sub", fontName=DEFAULT_FONT, fontSize=13, leading=16, textColor=colors.HexColor("#374151"), alignment=1)
INSIGHT_STYLE = _para_style("insight", fontName=DEFAULT_FONT, fontSize=9, leading=12, textColor=ACCENT_COLOR, italic=True)

def _format_dataframe_for_table(df):
    """
    Convert pandas DataFrame to reportlab-friendly list of lists with
    stringified values and header row.
    """
    try:
        # avoid importing pandas here; assume df implements .columns and .values
        df2 = df.copy()
        # round numeric columns if pandas available
        try:
            import pandas as pd
            for c in df2.select_dtypes(include=["float", "int"]).columns:
                df2[c] = df2[c].round(2)
        except Exception:
            pass

        # stringify all entries to avoid serialization issues
        cols = [str(c) for c in df2.columns]
        rows = [[str(x) for x in r] for r in df2.values.tolist()]
        return [cols] + rows
    except Exception:
        return None

def _safe_save_chart(title, fig):
    """
    Uses ensure_chart_saved (if available) or save_chart_image directly.
    Returns path or None.
    """
    try:
        if ensure_chart_saved:
            path = ensure_chart_saved(title, fig)
            if path:
                # tiny wait for filesystem
                time.sleep(0.08)
                return path
        # fallback
        if save_chart_image:
            path = save_chart_image(title, fig)
            if path:
                time.sleep(0.08)
                return path
    except Exception as e:
        # bubble up nothing to streamlit here (caller handles UI)
        return None
    return None

# Footer callback
def _add_footer(canvas, doc):
    canvas.saveState()
    footer_text = "Prepared with ❤️ by People Analytics Project — 2025"
    canvas.setFont(DEFAULT_FONT if DEFAULT_FONT else "Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    # center at bottom
    canvas.drawCentredString(PAGE_WIDTH / 2.0, 12 * mm, footer_text)
    canvas.restoreState()

# -------------------------------------------------------
# Main function to render consolidated PDF
# -------------------------------------------------------
def render_consolidated_pdf(report_title: str, modules: list, file_prefix: str = "Consolidated_HR_Report"):
    """
    Create a consolidated multi-module PDF.
    Args:
      - report_title: str (e.g., "Quarterly HR Leadership Deck")
      - modules: list of dicts, each dict:
           {
               "module_name": "Attrition",
               "module_desc": "Turnover & tenures",
               "data_blocks": [  # same shape as module-level data_blocks your modules use
                   {"title": "...", "desc": "...", "df": pandas.DataFrame or None, "fig": plotly_fig_or_path_or_None, "insights":[...]},
                   ...
               ]
           }
      - file_prefix: filename prefix for download button
    Behavior:
      - Builds cover, TOC, divider pages, section pages, summary, and returns a Streamlit download button.
    """
    if not isinstance(modules, list) or len(modules) == 0:
        st.warning("No modules provided for consolidated PDF.")
        return

    # Button triggers the build
    if not st.button("🧾 Generate Consolidated HR Executive PDF", use_container_width=True):
        return

    # buffer for PDF bytes
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=RIGHT_MARGIN,
        leftMargin=LEFT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
    )

    story = []

    # --------------- COVER PAGE ---------------
    story.append(Spacer(1, 60))
    story.append(Paragraph(f"<b>{report_title}</b>", COVER_TITLE))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"<para align=center><font size=12 color='#374151'>Consolidated HR Leadership Deck</font></para>",
        COVER_SUB
    ))
    story.append(Spacer(1, 18))
    story.append(Paragraph(f"<para align=center><font size=10>Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}</font></para>", BODY_STYLE))
    story.append(Spacer(1, 30))
    story.append(PageBreak())

    # ---------------- TABLE OF CONTENTS ----------------
    toc_data = [["#", "Section", "Description", "Page"]]
    page_counter = 2  # cover uses page 1, TOC considered page 2; sections will be assigned sequentially
    # compute approximate page numbers: each module may be multiple pages but we keep simple: module start page is incremental
    for idx, mod in enumerate(modules, start=1):
        toc_data.append([
            idx,
            mod.get("module_name", f"Module {idx}"),
            mod.get("module_desc", ""),
            str(page_counter)
        ])
        # assume each module will use at least 1 page + len(data_blocks) pages; rough increment to avoid TOC overflow
        blocks = mod.get("data_blocks", [])
        page_counter += max(1, len(blocks) + 1)

    story.append(Paragraph("<b>Table of Contents</b>", HEADING_STYLE))
    story.append(Spacer(1, 6))
    toc_table = Table(toc_data, colWidths=TOC_COLWIDTHS)
    toc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TOC_HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), DEFAULT_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # --------------- MODULES LOOP ---------------
    summary_rows = [["Section", "Key Insights"]]
    for midx, mod in enumerate(modules, start=1):
        module_name = mod.get("module_name", f"Module {midx}")
        module_desc = mod.get("module_desc", "")
        data_blocks = mod.get("data_blocks", []) or []

        # Divider page for module
        divider_style = ParagraphStyle(
            "divider",
            fontName=DEFAULT_FONT,
            fontSize=18,
            leading=22,
            alignment=1,
            textColor=colors.white,
            backColor=SECTION_COLOR,
            spaceBefore=8,
            spaceAfter=8,
        )
        # small divider - center title
        story.append(Paragraph(f"<para align=center><b>{module_name.upper()}</b></para>", divider_style))
        story.append(Spacer(1, 6))

        # For each data block in module
        for bidx, block in enumerate(data_blocks, start=1):
            title = block.get("title", f"Section {bidx}")
            desc = block.get("desc", "")
            df = block.get("df", None)
            fig = block.get("fig", None)
            insights = block.get("insights", []) or []

            story.append(Paragraph(f"{midx}.{bidx} {title}", HEADING_STYLE))
            if desc:
                story.append(Paragraph(desc, BODY_STYLE))
            story.append(Spacer(1, 6))

            # Table
            if df is not None:
                try:
                    table_data = _format_dataframe_for_table(df)
                    if table_data:
                        col_count = len(table_data[0])
                        col_width = (PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN) / col_count
                        tb = Table(table_data, colWidths=[col_width] * col_count, repeatRows=1)
                        tb.setStyle(TableStyle([
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                            ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]))
                        story.append(tb)
                        story.append(Spacer(1, 8))
                except Exception as e:
                    story.append(Paragraph(f"⚠️ Table render error: {e}", BODY_STYLE))
                    story.append(Spacer(1, 8))

            # Chart
            if fig is not None:
                try:
                    img_path = None
                    # If fig already a path string, use it; else attempt to save via saver
                    if isinstance(fig, str) and os.path.exists(fig):
                        img_path = fig
                    else:
                        img_path = _safe_save_chart(f"{module_name}_{title}", fig)

                    # wait a small amount for fs
                    if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                        # place image scaled to target mm
                        try:
                            rl_img = RLImage(img_path, width=CHART_W_MM * mm, height=CHART_H_MM * mm)
                            story.append(rl_img)
                            story.append(Spacer(1, 8))
                        except Exception as e:
                            story.append(Paragraph(f"⚠️ Chart embed error: {e}", BODY_STYLE))
                    else:
                        story.append(Paragraph("⚠️ Chart could not be rendered.", BODY_STYLE))
                except Exception as e:
                    story.append(Paragraph(f"⚠️ Chart render error: {e}", BODY_STYLE))
            else:
                story.append(Paragraph("⚠️ No chart available for this section.", BODY_STYLE))

            # Insights
            if insights:
                try:
                    insights_text = " • ".join([str(x) for x in insights])
                    story.append(Spacer(1, 4))
                    story.append(Paragraph(f"<i>{insights_text}</i>", INSIGHT_STYLE))
                except Exception:
                    pass

            # add to summary table
            summary_rows.append([f"{module_name} — {title}", " • ".join([str(x) for x in insights])])
            story.append(PageBreak())

    # --------------- EXECUTIVE SUMMARY ---------------
    story.append(Paragraph("Executive Summary", HEADING_STYLE))
    story.append(Spacer(1, 8))
    try:
        summary_table = Table(summary_rows, colWidths=[140, 310])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TOC_HEADER_BG),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, 0), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 8))
    except Exception as e:
        story.append(Paragraph(f"⚠️ Executive summary build failed: {e}", BODY_STYLE))

    # finally build document with footer
    try:
        doc.build(story, onLaterPages=_add_footer)
        pdf_bytes = buf.getvalue()
        st.success("✅ Consolidated Executive PDF generated successfully.")
        st.download_button(
            "⬇️ Download Consolidated Report",
            pdf_bytes,
            file_name=f"{file_prefix}_Executive_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"⚠️ PDF build failed: {e}")
        st.write(traceback.format_exc())
    finally:
        buf.close()