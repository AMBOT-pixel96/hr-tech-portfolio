# ============================================
# utils/pdf_helper.py — v4.0 | PDF Builder (uses chart_saver v4.0)
# ============================================
import os
import io
import time
import math
import streamlit as st
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import pandas as pd

# use chart_saver.ensure_chart_saved for robust image creation
from utils.chart_saver import ensure_chart_saved

# Font registration: look in repo utils/fonts first
FONT_FILE = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
DEFAULT_FONT = "Helvetica"
if os.path.exists(FONT_FILE):
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_FILE))
        DEFAULT_FONT = "DejaVuSans"
    except Exception:
        DEFAULT_FONT = "Helvetica"

# Page constants
PAGE_W, PAGE_H = A4
MARGIN_LR = 18 * mm
MARGIN_TB = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_LR
IMG_MAX_W = CONTENT_W
IMG_MAX_H = 110 * mm

def build_toc_tables(data_blocks, row_limit=18):
    rows = [["#", "Section", "Description", "Page"]]
    for i, block in enumerate(data_blocks, 1):
        desc = block.get("desc", "") or ""
        # soft truncate long descriptions to avoid giant rows; preserve words
        if len(desc) > 140:
            desc = desc[:137] + "..."
        rows.append([str(i), block.get("title", ""), desc, str(i + 1)])
    tables = []
    header = rows[0]
    data_rows = rows[1:]
    for i in range(0, len(data_rows), row_limit):
        chunk = data_rows[i: i + row_limit]
        table_data = [header] + chunk
        tbl = Table(table_data, colWidths=[30, 120, CONTENT_W - (30 + 40), 40])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 1), (0, -1), "CENTER"),
            ("ALIGN", (3, 1), (3, -1), "RIGHT"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ]))
        tables.append(tbl)
    return tables

def compute_image_size(img_path, max_w=IMG_MAX_W, max_h=IMG_MAX_H):
    # conservative embedding: use PIL if available to get aspect ratio
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            w_px, h_px = im.size
            aspect = w_px / float(h_px) if h_px else 1.0
            target_w = min(max_w, float(w_px) * 0.6)
            target_h = target_w / aspect
            if target_h > max_h:
                target_h = max_h
                target_w = target_h * aspect
            return float(target_w), float(target_h)
    except Exception:
        return float(max_w), float(max_h)

def render_pdf_download_button(report_title, module_name, data_blocks, file_prefix):
    """
    Generate polished executive PDF using in-memory chart export via chart_saver.ensure_chart_saved.
    """
    if not data_blocks:
        st.warning("⚠️ No data blocks available.")
        return

    if not st.button(f"🧾 Generate {module_name} Executive PDF", use_container_width=True):
        return

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=MARGIN_LR, leftMargin=MARGIN_LR,
        topMargin=MARGIN_TB, bottomMargin=MARGIN_TB
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"],
                          fontName=DEFAULT_FONT, fontSize=10, leading=13, textColor=colors.black)
    heading = ParagraphStyle("heading", parent=styles["Heading2"],
                             fontName=DEFAULT_FONT, fontSize=13, textColor=colors.HexColor("#1E3A8A"))

    story = []

    # COVER
    story.append(Spacer(1, 70))
    story.append(Paragraph(f"<para align=center><font size=24><b>{report_title}</b></font></para>", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<para align=center><font size=14 color='#374151'>{module_name} Module</font></para>", body))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<para align=center><font size=10>Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}</font></para>", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<para align=center><font size=9 color='#6B7280'>Prepared by Amlan Mishra | © 2025 HR Tech Portfolio</font></para>", body))
    story.append(PageBreak())

    # TOC
    toc_tables = build_toc_tables(data_blocks, row_limit=18)
    story.append(Paragraph("<b>Table of Contents</b>", heading))
    story.append(Spacer(1, 6))
    for ti, t in enumerate(toc_tables):
        story.append(t)
        if ti != len(toc_tables) - 1:
            story.append(PageBreak())
    story.append(PageBreak())

    # SECTIONS
    summary_rows = [["Section", "Key Insights"]]

    for idx, block in enumerate(data_blocks, 1):
        title = block.get("title", f"Section {idx}")
        desc = block.get("desc", "")
        df = block.get("df", None)
        fig = block.get("fig", None)
        insights = block.get("insights", []) or []

        # header
        story.append(KeepTogether([
            Paragraph(f"{idx}. {title}", heading),
            Paragraph(desc or "", body),
            Spacer(1, 6)
        ]))

        # Table rendering (summary only; cap rows)
        if df is not None:
            try:
                # cap to top N rows to avoid huge employee dumps
                max_rows_shown = 30
                df2 = df.copy()
                # round numeric values to 2 decimals for readability
                for c in df2.select_dtypes(include=["float", "int"]).columns:
                    df2[c] = df2[c].round(2)
                if len(df2) > max_rows_shown:
                    df_pdf = df2.head(max_rows_shown)
                    # add ellipsis row
                    ell = {col: "..." for col in df_pdf.columns}
                    df_pdf = pd.concat([df_pdf, pd.DataFrame([ell])], ignore_index=True)
                else:
                    df_pdf = df2

                table_data = [list(df_pdf.columns)] + df_pdf.fillna("").astype(str).values.tolist()
                col_count = max(1, len(table_data[0]))
                colw = (PAGE_W - 2 * MARGIN_LR) / col_count
                tbl = Table(table_data, colWidths=[colw] * col_count, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                    ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 8))
            except Exception as e:
                story.append(Paragraph(f"⚠️ Table render error: {e}", body))
        else:
            story.append(Paragraph("⚠️ No table available for this section.", body))
            story.append(Spacer(1, 6))

        # Chart embedding: use ensure_chart_saved for robust save
        img_path = None
        if fig is not None:
            try:
                img_path = ensure_chart_saved(title, fig)
                if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    w_pt, h_pt = compute_image_size(img_path, max_w=IMG_MAX_W, max_h=IMG_MAX_H)
                    story.append(RLImage(img_path, width=w_pt, height=h_pt))
                    story.append(Spacer(1, 8))
                else:
                    story.append(Paragraph("⚠️ Chart could not be rendered.", body))
            except Exception as e:
                story.append(Paragraph(f"⚠️ Chart render error: {e}", body))
        else:
            story.append(Paragraph("⚠️ No chart available for this section.", body))

        # Insights
        try:
            if insights:
                insight_text = " • ".join(map(str, insights))
                story.append(Spacer(1, 4))
                story.append(Paragraph(f"<font color='#2563EB'><i>{insight_text}</i></font>", body))
            else:
                story.append(Spacer(1, 2))
        except Exception:
            story.append(Paragraph("• " + " ".join(map(str, insights)), body))

        summary_rows.append([title, " • ".join(map(str, insights))])
        story.append(PageBreak())

    # SUMMARY
    story.append(Paragraph("Executive Summary", heading))
    story.append(Spacer(1, 6))
    try:
        summary_table = Table(summary_rows, colWidths=[140, PAGE_W - 2 * MARGIN_LR - 140])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(summary_table)
    except Exception as e:
        story.append(Paragraph(f"⚠️ Summary render error: {e}", body))

    # BUILD & DOWNLOAD
    try:
        doc.build(story)
        pdf_bytes = buf.getvalue()
        st.success("✅ Executive PDF generated successfully.")
        st.download_button("⬇️ Download Report", pdf_bytes, file_name=f"{file_prefix}_Executive_Report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"⚠️ PDF build failed: {e}")
    finally:
        try:
            buf.close()
        except Exception:
            pass