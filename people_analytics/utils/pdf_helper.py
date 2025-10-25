# ============================================
# utils/pdf_helper.py — v5.0 | Executive Polish Edition (Stable)
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
from utils.chart_saver import save_chart_image, ensure_chart_saved

# ✅ Register Unicode font for ₹ and all symbols
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
except:
    pass
DEFAULT_FONT = "DejaVuSans"


# =====================================================
# 🧩 Main Export Function
# =====================================================
def render_pdf_download_button(report_title, module_name, data_blocks, file_prefix):
    """Builds a clean, full-color executive PDF report."""
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

        # -------------------------------------------------
        # Styles
        # -------------------------------------------------
        styles = getSampleStyleSheet()
        body = ParagraphStyle(
            "body",
            parent=styles["Normal"],
            fontName=DEFAULT_FONT,
            fontSize=10,
            leading=14,
            textColor=colors.black,
        )
        heading = ParagraphStyle(
            "heading",
            parent=styles["Heading2"],
            fontName=DEFAULT_FONT,
            fontSize=13,
            textColor=colors.HexColor("#1E3A8A"),
            spaceAfter=6,
        )

        story = []

        # -------------------------------------------------
        # 🧠 COVER PAGE
        # -------------------------------------------------
        story.append(Spacer(1, 100))
        story.append(Paragraph(
            f"<para align=center><font size=22><b>{report_title}</b></font></para>", body))
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            f"<para align=center><font size=13 color='#374151'>{module_name} Module</font></para>", body))
        story.append(Spacer(1, 40))
        story.append(Paragraph(
            f"<para align=center><font size=10>Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}</font></para>", body))
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
        toc_table = Table(
            toc_data,
            colWidths=[20 * mm, 50 * mm, 95 * mm, 15 * mm],
            repeatRows=1
        )
        toc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("WORDWRAP", (1, 1), (-2, -1), "CJK"),
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
                col_width = (A4[0] - 60) / col_count
                table = Table(
                    table_data,
                    colWidths=[col_width] * col_count,
                    repeatRows=1
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
                    img_path = ensure_chart_saved(title, fig)
                    if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                        story.append(RLImage(img_path, width=175 * mm, height=105 * mm))
                        story.append(Spacer(1, 8))
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
        story.append(Paragraph("<b>Executive Summary</b>", heading))
        story.append(Spacer(1, 8))
        summary_table = Table(summary_data, colWidths=[70 * mm, 110 * mm], repeatRows=1)
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 10))

        # -------------------------------
        # 💾 BUILD & DOWNLOAD
        # -------------------------------
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

            # --------------------------------
            # 🧩 Auto-save for Consolidated Deck
            # --------------------------------
            try:
                from utils_consolidated.pdf_merger import TMP_DIR
                os.makedirs(TMP_DIR, exist_ok=True)
                pdf_save_path = os.path.join(TMP_DIR, f"{module_name}.pdf")
                with open(pdf_save_path, "wb") as f:
                    f.write(pdf_data)
                st.info("🧩 A copy of this report has been added to the consolidated deck queue.")
            except Exception as e:
                st.warning(f"⚠️ Could not auto-save PDF for consolidation: {e}")

        except Exception as e:
            st.error(f"⚠️ PDF build failed: {e}")
        finally:
            buf.close()