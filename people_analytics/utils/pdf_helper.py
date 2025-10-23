# ============================================
# utils/pdf_helper.py — v3.9 | Hybrid Renderer Edition
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

# Local helper that writes Plotly figs to PNG using Kaleido (or returns existing path)
from utils.chart_saver import save_chart_image

# Pillow for PNG post-processing (remove alpha, force white bg)
try:
    from PIL import Image
except Exception:
    Image = None

# ----------------------------
# Font registration (DejaVuSans recommended)
# ----------------------------
FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # typical Linux
    os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf"),
]

DEFAULT_FONT = "Helvetica"
for fp in FONT_PATH_CANDIDATES:
    try:
        if fp and os.path.exists(fp):
            pdfmetrics.registerFont(TTFont("DejaVuSans", fp))
            DEFAULT_FONT = "DejaVuSans"
            break
    except Exception:
        continue

# Page constants
PAGE_W, PAGE_H = A4
MARGIN_LR = 18 * mm
MARGIN_TB = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_LR
IMG_MAX_W = CONTENT_W
IMG_MAX_H = 110 * mm

# Helper: sanitize and enforce white bg for PNG using Pillow (in-place)
def sanitize_png_white_bg(img_path):
    """
    Open image at img_path and ensure it's RGB with a white background
    (removes transparency, preserves colors). Overwrites same path.
    """
    if Image is None:
        # Pillow not available — skip but warn
        st.warning("⚠️ Pillow not available - PNG post-processing skipped.")
        return img_path

    try:
        with Image.open(img_path) as im:
            # If image already RGB and no alpha channel, keep as is
            if im.mode in ("RGB", "L"):
                # convert to RGB just in case
                rgb = im.convert("RGB")
                rgb.save(img_path, format="PNG")
                return img_path

            # If has alpha (RGBA, LA), composite over white background
            if "A" in im.getbands() or im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                # if RGBA
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                bg.paste(im, mask=im.split()[-1])  # alpha channel as mask
                bg.save(img_path, format="PNG")
                return img_path

            # Fallback: convert to RGB
            im.convert("RGB").save(img_path, format="PNG")
            return img_path
    except Exception as e:
        st.warning(f"⚠️ PNG sanitization failed for {img_path}: {e}")
        return img_path

# Helper: smart image scaling while maintaining aspect ratio
def compute_image_size(img_path, max_w=IMG_MAX_W, max_h=IMG_MAX_H):
    if Image is None:
        # fallback approximations
        return (max_w, max_h)
    try:
        with Image.open(img_path) as im:
            w_px, h_px = im.size
            # convert px to points (ReportLab uses points; DPI assumption 96)
            # But easier: scale proportionally based on ratio of desired widths in points
            # We'll compute aspect = w/h and scale by max width/height
            aspect = w_px / float(h_px) if h_px else 1
            # initial candidate
            target_w = min(max_w, w_px * 0.75)
            target_h = target_w / aspect
            if target_h > max_h:
                target_h = max_h
                target_w = target_h * aspect
            return (float(target_w), float(target_h))
    except Exception:
        return (max_w, max_h)

# Helper: split TOC into multiple pages if too many rows to avoid overflow
def build_toc_tables(data_blocks, row_limit=18):
    """
    Returns a list of ReportLab Table objects for TOC; splits when rows exceed row_limit.
    row_limit is number of data rows (excluding header) per table.
    """
    rows = [["#", "Section", "Description", "Page"]]
    for i, block in enumerate(data_blocks, 1):
        rows.append([str(i), block.get("title", ""), block.get("desc", ""), str(i + 1)])
    tables = []
    header = rows[0]
    data_rows = rows[1:]
    # chunk data_rows
    for i in range(0, len(data_rows), row_limit):
        chunk = data_rows[i : i + row_limit]
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

# Main export function
def render_pdf_download_button(report_title, module_name, data_blocks, file_prefix):
    """
    Generate a polished, color-safe A4 Executive report PDF and show a Streamlit download button.
    """
    if not data_blocks:
        st.warning("⚠️ No data blocks available.")
        return

    # the generation button (keeps user explicit)
    if not st.button(f"🧾 Generate {module_name} Executive PDF", use_container_width=True):
        return

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=MARGIN_LR, leftMargin=MARGIN_LR,
        topMargin=MARGIN_TB, bottomMargin=MARGIN_TB,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"],
                          fontName=DEFAULT_FONT, fontSize=10,
                          leading=13, textColor=colors.black)
    heading = ParagraphStyle("heading", parent=styles["Heading2"],
                             fontName=DEFAULT_FONT, fontSize=13,
                             textColor=colors.HexColor("#1E3A8A"))

    story = []

    # ---- COVER PAGE (locked layout & consistent spacing) ----
    story.append(Spacer(1, 70))
    story.append(Paragraph(f"<para align=center><font size=24><b>{report_title}</b></font></para>", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<para align=center><font size=14 color='#374151'>{module_name} Module</font></para>", body))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<para align=center><font size=10>Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}</font></para>", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<para align=center><font size=9 color='#6B7280'>Prepared by Amlan Mishra | © 2025 HR Tech Portfolio</font></para>", body))
    story.append(PageBreak())

    # ---- TABLE OF CONTENTS (split into pages if large) ----
    toc_tables = build_toc_tables(data_blocks, row_limit=18)
    story.append(Paragraph("<b>Table of Contents</b>", heading))
    story.append(Spacer(1, 6))
    for t_idx, t in enumerate(toc_tables):
        story.append(t)
        # insert page break if more than one TOC-table
        if t_idx != len(toc_tables) - 1:
            story.append(PageBreak())
    story.append(PageBreak())

    # ---- SECTIONS: Metric -> Table -> Chart -> Insight (one per page) ----
    summary_rows = [["Section", "Key Insights"]]

    for idx, block in enumerate(data_blocks, 1):
        title = block.get("title", f"Section {idx}")
        desc = block.get("desc", "")
        df = block.get("df")
        fig = block.get("fig")
        insights = block.get("insights", []) or []

        # Title + Desc wrapped inside KeepTogether to avoid split heading
        story.append(KeepTogether([
            Paragraph(f"{idx}. {title}", heading),
            Paragraph(desc or "", body),
            Spacer(1, 6)
        ]))

        # Table (summary only) — convert numeric columns to two decimals and truncate long tables
        if df is not None:
            try:
                # prefer showing summary (not full employee dump). If many rows, show top N
                max_rows_shown = 30
                df2 = df.copy()
                # round numeric to 2 decimals safely
                for c in df2.select_dtypes(include=["float", "int"]).columns:
                    df2[c] = df2[c].round(2)
                if len(df2) > max_rows_shown:
                    df_for_pdf = pd_head = df2.head(max_rows_shown)
                    # add a footnote row
                    foot = ["..."] * len(df_for_pdf.columns)
                    df_for_pdf.loc[len(df_for_pdf)] = foot
                else:
                    df_for_pdf = df2
                # convert all to strings for robust Table creation
                pdf_table_data = [list(df_for_pdf.columns)] + df_for_pdf.fillna("").astype(str).values.tolist()
                col_count = len(pdf_table_data[0]) if pdf_table_data else 1
                cw = (PAGE_W - 2 * MARGIN_LR) / max(1, col_count)
                tbl = Table(pdf_table_data, colWidths=[cw] * col_count, repeatRows=1)
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

        # Chart embedding: attempt to save with save_chart_image, sanitize PNG, then embed
        img_path = None
        if fig is not None:
            try:
                # If fig is already a path (string), accept it
                if isinstance(fig, str) and os.path.exists(fig):
                    img_path = fig
                else:
                    img_path = save_chart_image(title, fig)

                # sanitize PNG so transparency -> white, preserve colors
                if img_path:
                    img_path = sanitize_png_white_bg(img_path)

                # ensure file exists & has size
                wait_attempts = 6
                for _ in range(wait_attempts):
                    if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                        break
                    time.sleep(0.25)

                if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    # compute safe sized embedding
                    w_pt, h_pt = compute_image_size(img_path, max_w=IMG_MAX_W, max_h=IMG_MAX_H)
                    story.append(RLImage(img_path, width=w_pt, height=h_pt))
                    story.append(Spacer(1, 8))
                else:
                    story.append(Paragraph("⚠️ Chart could not be rendered.", body))
            except Exception as e:
                story.append(Paragraph(f"⚠️ Chart render error: {e}", body))
        else:
            story.append(Paragraph("⚠️ No chart available for this section.", body))

        # Insights (bulletline)
        if insights:
            try:
                insight_text = " • ".join(map(str, insights))
                story.append(Spacer(1, 4))
                story.append(Paragraph(f"<font color='#2563EB'><i>{insight_text}</i></font>", body))
            except Exception:
                story.append(Paragraph("• " + " ".join(map(str, insights)), body))

        summary_rows.append([title, " • ".join(map(str, insights))])
        story.append(PageBreak())

    # Executive Summary page
    story.append(Paragraph("Executive Summary", heading))
    story.append(Spacer(1, 6))
    # Build summary table (safe widths)
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

    # Build document
    try:
        doc.build(story)
        pdf_bytes = buf.getvalue()
        st.success("✅ Executive PDF generated successfully.")
        st.download_button(
            "⬇️ Download Report",
            pdf_bytes,
            file_name=f"{file_prefix}_Executive_Report.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"⚠️ PDF build failed: {e}")
    finally:
        try:
            buf.close()
        except Exception:
            pass