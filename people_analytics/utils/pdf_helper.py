# ============================================
# utils/pdf_helper.py — v3.8 | Final Executive Layout (Color-Safe & Stable)
# ============================================
import os, io, time, streamlit as st
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from utils.chart_saver import save_chart_image

# --------------------------------------------
# ✅ Register font (handles ₹)
# --------------------------------------------
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = "Helvetica"

PAGE_W, PAGE_H = A4
MARGIN_LR = 18 * mm
MARGIN_TB = 20 * mm
IMG_MAX_W = PAGE_W - 2 * MARGIN_LR
IMG_MAX_H = 110 * mm  # auto scale target

# =====================================================
# 🧩 Main Export Function
# =====================================================
def render_pdf_download_button(report_title, module_name, data_blocks, file_prefix):
    if not data_blocks:
        st.warning("⚠️ No data blocks available.")
        return

    if st.button(f"🧾 Generate {module_name} Executive PDF", use_container_width=True):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=MARGIN_LR, leftMargin=MARGIN_LR,
            topMargin=MARGIN_TB, bottomMargin=MARGIN_TB
        )

        styles = getSampleStyleSheet()
        body = ParagraphStyle("body", parent=styles["Normal"],
                              fontName=DEFAULT_FONT, fontSize=10,
                              leading=13, textColor=colors.black)
        heading = ParagraphStyle("heading", parent=styles["Heading2"],
                                 fontName=DEFAULT_FONT, fontSize=13,
                                 textColor=colors.HexColor("#1E3A8A"))

        story = []

        # -------------------------------------------------
        # 📘 COVER PAGE
        # -------------------------------------------------
        story += [
            Spacer(1, 70),
            Paragraph(f"<para align=center><font size=24><b>{report_title}</b></font></para>", body),
            Spacer(1, 10),
            Paragraph(f"<para align=center><font size=14 color='#374151'>{module_name} Module</font></para>", body),
            Spacer(1, 40),
            Paragraph(f"<para align=center><font size=10>Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}</font></para>", body),
            Spacer(1, 20),
            Paragraph("<para align=center><font size=9 color='#6B7280'>Prepared by Amlan Mishra | © 2025 HR Tech Portfolio</font></para>", body),
            PageBreak()
        ]

        # -------------------------------------------------
        # 📖 TABLE OF CONTENTS
        # -------------------------------------------------
        toc_rows = [["#", "Section", "Description", "Page"]]
        for i, block in enumerate(data_blocks, 1):
            toc_rows.append([i, block.get("title", ""), block.get("desc", ""), str(i + 1)])

        toc_table = Table(toc_rows, colWidths=[20, 120, 220, 30])
        toc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (3, 1), (3, -1), "RIGHT"),
        ]))
        story += [Paragraph("<b>Table of Contents</b>", heading), Spacer(1, 6), toc_table, PageBreak()]

        # -------------------------------------------------
        # 📊 SECTIONS
        # -------------------------------------------------
        summary_rows = [["Section", "Key Insights"]]

        for i, block in enumerate(data_blocks, 1):
            title = block.get("title", f"Section {i}")
            desc = block.get("desc", "")
            df = block.get("df")
            fig = block.get("fig")
            insights = block.get("insights", [])

            story.append(Paragraph(f"{i}. {title}", heading))
            if desc:
                story.append(Paragraph(desc, body))
            story.append(Spacer(1, 6))

            # TABLE
            if df is not None and not df.empty:
                df = df.round(2).astype(str)
                table_data = [list(df.columns)] + df.values.tolist()
                cw = (PAGE_W - 2 * MARGIN_LR) / len(df.columns)
                t = Table(table_data, colWidths=[cw] * len(df.columns), repeatRows=1)
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                    ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))

            # CHART
            img_path = None
            if fig is not None:
                try:
                    if isinstance(fig, str) and os.path.exists(fig):
                        img_path = fig
                    else:
                        img_path = save_chart_image(title, fig)

                    if img_path and os.path.exists(img_path):
                        # --- get real size to scale ---
                        from PIL import Image
                        with Image.open(img_path) as im:
                            w, h = im.size
                        aspect = w / h
                        new_w = min(IMG_MAX_W, w * 0.6)
                        new_h = min(IMG_MAX_H, new_w / aspect)
                        story.append(RLImage(img_path, width=new_w, height=new_h))
                        story.append(Spacer(1, 10))
                    else:
                        story.append(Paragraph("⚠️ Chart not available.", body))
                except Exception as e:
                    story.append(Paragraph(f"⚠️ Chart render error: {e}", body))
            else:
                story.append(Paragraph("⚠️ No chart available for this section.", body))

            # INSIGHTS
            if insights:
                txt = " • ".join(map(str, insights))
                story.append(Spacer(1, 4))
                story.append(Paragraph(f"<font color='#2563EB'><i>{txt}</i></font>", body))

            summary_rows.append([title, " • ".join(map(str, insights))])
            story.append(PageBreak())

        # -------------------------------------------------
        # 🧠 EXECUTIVE SUMMARY
        # -------------------------------------------------
        story.append(Paragraph("Executive Summary", heading))
        story.append(Spacer(1, 6))
        summary_table = Table(summary_rows, colWidths=[140, 310])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(summary_table)

        # -------------------------------------------------
        # 💾 BUILD & DOWNLOAD
        # -------------------------------------------------
        try:
            doc.build(story)
            st.success("✅ Executive PDF generated successfully.")
            st.download_button(
                "⬇️ Download Report",
                buf.getvalue(),
                file_name=f"{file_prefix}_Executive_Report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"⚠️ PDF build failed: {e}")
        finally:
            buf.close()