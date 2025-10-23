# ============================================
# utils/pdf_helper.py — v3.6 | Executive Color Deck (FINAL)
# ============================================
import os
import io
import time
import streamlit as st
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from utils.chart_saver import save_chart_image

# ✅ Register Unicode Font (fixes ₹ and symbols)
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
except:
    pass
DEFAULT_FONT = "DejaVuSans"

# =====================================================
# 🧩 Main Export Function
# =====================================================
def render_pdf_download_button(report_title, module_name, data_blocks, file_prefix):
    """
    Generate a polished, full-color A4 Executive Report.
    ✅ Unicode-safe (₹ symbol)
    ✅ Full-width charts with consistent scaling
    ✅ Proper TOC wrapping, spacing, and color retention
    """
    if not data_blocks:
        st.warning("⚠️ No data blocks available.")
        return

    if st.button(f"🧾 Generate {module_name} Executive PDF", use_container_width=True):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )

        # Styles
        styles = getSampleStyleSheet()
        body = ParagraphStyle(
            "body",
            parent=styles["Normal"],
            fontName=DEFAULT_FONT,
            fontSize=10,
            leading=13,
            textColor=colors.black,
        )
        heading = ParagraphStyle(
            "heading",
            parent=styles["Heading2"],
            fontName=DEFAULT_FONT,
            fontSize=13,
            textColor=colors.HexColor("#1E3A8A"),
        )

        story = []

        # -------------------------------------------------
        # 🧠 COVER PAGE
        # -------------------------------------------------
        story.append(Spacer(1, 70))
        story.append(Paragraph(
            f"<para align=center><font size=24><b>{report_title}</b></font></para>", body))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"<para align=center><font size=14 color='#374151'>{module_name} Module</font></para>", body))
        story.append(Spacer(1, 40))
        story.append(Paragraph(
            f"<para align=center><font size=11>Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}</font></para>", body))
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            "<para align=center><font size=9 color='#6B7280'>Prepared by Amlan Mishra | © 2025 HR Tech Portfolio</font></para>", body))
        story.append(PageBreak())

        # -------------------------------------------------
        # 📖 TABLE OF CONTENTS
        # -------------------------------------------------
        toc_data = [["#", "Section", "Description", "Page"]]
        for i, block in enumerate(data_blocks, 1):
            toc_data.append([
                i,
                block.get("title", ""),
                block.get("desc", ""),
                str(i + 1)
            ])
        toc_table = Table(toc_data, colWidths=[20, 90, 250, 30], repeatRows=1)
        toc_table._argW[2] = 250  # enable wrapping
        toc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(Paragraph("<b>Table of Contents</b>", heading))
        story.append(Spacer(1, 6))
        story.append(toc_table)
        story.append(PageBreak())

        # -------------------------------------------------
        # 📊 SECTION LOOP
        # -------------------------------------------------
        summary_data = [["Section", "Key Insights"]]

        for i, block in enumerate(data_blocks, 1):
            title = block.get("title", f"Section {i}")
            desc = block.get("desc", "")
            df = block.get("df", None)
            fig = block.get("fig", None)
            insights = block.get("insights", [])

            story.append(Paragraph(f"{i}. {title}", heading))
            story.append(Paragraph(desc, body))
            story.append(Spacer(1, 6))

            # --- TABLE ---
            if df is not None and not df.empty:
                df = df.round(2).astype(str)
                table_data = [list(df.columns)] + df.values.tolist()
                col_count = len(df.columns)
                table = Table(
                    table_data,
                    colWidths=[(A4[0] - 40) / col_count] * col_count,
                    repeatRows=1,
                )
                table.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                    ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(table)
                story.append(Spacer(1, 10))

            # --- CHART ---
            img_path = None
            if fig is not None:
                try:
                    if isinstance(fig, str) and os.path.exists(fig):
                        img_path = fig
                    else:
                        img_path = save_chart_image(title, fig)

                    # Wait for file flush
                    for _ in range(5):
                        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                            break
                        time.sleep(0.3)

                    if img_path and os.path.exists(img_path):
                        story.append(RLImage(img_path, width=180 * mm, height=105 * mm))
                        story.append(Spacer(1, 10))
                    else:
                        story.append(Paragraph("⚠️ Chart could not be rendered.", body))
                except Exception as e:
                    story.append(Paragraph(f"⚠️ Chart render error: {e}", body))
            else:
                story.append(Paragraph("⚠️ No chart available for this section.", body))

            # --- INSIGHTS ---
            if insights:
                insights_text = " • ".join(str(x) for x in insights)
                story.append(Spacer(1, 4))
                story.append(Paragraph(
                    f"<font color='#2563EB'><i>{insights_text}</i></font>", body))
            else:
                insights_text = ""

            summary_data.append([title, insights_text])
            story.append(PageBreak())

        # -------------------------------------------------
        # 📘 EXECUTIVE SUMMARY
        # -------------------------------------------------
        story.append(Paragraph("Executive Summary", heading))
        story.append(Spacer(1, 8))
        summary_table = Table(summary_data, colWidths=[140, 310])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 10))

        # -------------------------------------------------
        # 💾 BUILD & DOWNLOAD
        # -------------------------------------------------
        try:
            doc.build(story)
            pdf_data = buf.getvalue()
            st.success("✅ Executive PDF generated successfully.")
            st.download_button(
                "⬇️ Download Report",
                pdf_data,
                file_name=f"{file_prefix}_Executive_Report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"⚠️ PDF build failed: {e}")
        finally:
            buf.close()