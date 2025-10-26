# ============================================
# utils/pdf_helper.py — v5.2 | Executive Standalone Edition (Restored)
# ============================================
import os
import io
import streamlit as st
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from utils.chart_saver import ensure_chart_saved

try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
except:
    pass
DEFAULT_FONT = "DejaVuSans"


def render_pdf_download_button(report_title, module_name, data_blocks, file_prefix):
    """Generate a full standalone executive PDF report (with timestamp, tables, visuals, summary)."""
    if not data_blocks:
        st.warning("⚠️ No data blocks available.")
        return

    if st.button(f"🧾 Generate {module_name} Executive PDF", use_container_width=True):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=18 * mm, leftMargin=18 * mm,
            topMargin=20 * mm, bottomMargin=20 * mm
        )

        styles = getSampleStyleSheet()
        body = ParagraphStyle("body", parent=styles["Normal"], fontName=DEFAULT_FONT, fontSize=10, leading=14)
        heading = ParagraphStyle("heading", parent=styles["Heading2"], fontName=DEFAULT_FONT,
                                 fontSize=13, textColor=colors.HexColor("#1E3A8A"), spaceAfter=6)
        story = []

        # -------------------------------------------------
        # COVER PAGE
        # -------------------------------------------------
        story.append(Spacer(1, 100))
        story.append(Paragraph(f"<para align=center><font size=22><b>{report_title}</b></font></para>", body))
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            f"<para align=center><font size=13 color='#374151'>{module_name} Module</font></para>", body))
        story.append(Spacer(1, 40))
        story.append(Paragraph(
            f"<para align=center><font size=10>Generated on {datetime.now().strftime('%d-%b-%Y')}</font></para>", body))
        story.append(PageBreak())

        # -------------------------------------------------
        # TABLE OF CONTENTS (with page numbers for standalone)
        # -------------------------------------------------
        toc_data = [["#", "Section", "Description", "Page"]]
        for i, block in enumerate(data_blocks, 1):
            toc_data.append([i, block.get("title", ""), block.get("desc", ""), str(i + 1)])
        toc_table = Table(toc_data, colWidths=[20*mm, 55*mm, 85*mm, 15*mm])
        toc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9)
        ]))
        story.append(Paragraph("<b>Table of Contents</b>", heading))
        story.append(Spacer(1, 8))
        story.append(toc_table)
        story.append(PageBreak())

        # -------------------------------------------------
        # SECTIONS
        # -------------------------------------------------
        summary_data = [["Section", "Key Insights"]]
        for i, block in enumerate(data_blocks, 1):
            title, desc, df, fig, insights = (
                block.get("title", f"Section {i}"),
                block.get("desc", ""),
                block.get("df", None),
                block.get("fig", None),
                block.get("insights", []),
            )

            story.append(Paragraph(f"{i}. {title}", heading))
            story.append(Paragraph(desc, body))
            story.append(Spacer(1, 6))

            if df is not None and not df.empty:
                df = df.round(2).astype(str)
                table_data = [list(df.columns)] + df.values.tolist()
                col_count = len(df.columns)
                table = Table(table_data, colWidths=[(A4[0]-60)/col_count]*col_count, repeatRows=1)
                table.setStyle(TableStyle([
                    ("GRID", (0,0), (-1,-1), 0.25, colors.black),
                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F3F4F6")),
                    ("FONTNAME", (0,0), (-1,-1), DEFAULT_FONT),
                    ("FONTSIZE", (0,0), (-1,-1), 9)
                ]))
                story.append(table)
                story.append(Spacer(1, 10))

            if fig is not None:
                img = ensure_chart_saved(title, fig)
                if img and os.path.exists(img):
                    story.append(RLImage(img, width=175*mm, height=105*mm))
                    story.append(Spacer(1, 8))

            if insights:
                story.append(Paragraph(" • ".join(insights), body))
            summary_data.append([title, " • ".join(insights)])
            story.append(PageBreak())

        # -------------------------------------------------
        # EXECUTIVE SUMMARY
        # -------------------------------------------------
        story.append(Paragraph("<b>Executive Summary</b>", heading))
        story.append(Spacer(1, 8))
        summary_table = Table(summary_data, colWidths=[70*mm, 110*mm])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), DEFAULT_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 9)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 10))

        # -------------------------------------------------
        # BUILD + SAVE
        # -------------------------------------------------
        try:
            doc.build(story)
            pdf_data = buf.getvalue()
            st.success("✅ Executive PDF generated successfully.")
            st.download_button("⬇️ Download Report", pdf_data, file_name=f"{file_prefix}_Executive_Report.pdf", mime="application/pdf")

            # Save copy for consolidated
            from utils_consolidated.pdf_merger import TMP_DIR
            os.makedirs(TMP_DIR, exist_ok=True)
            with open(os.path.join(TMP_DIR, f"{module_name}.pdf"), "wb") as f:
                f.write(pdf_data)

            # Save insights metadata
            import json
            insights_joined = " • ".join([ins for b in data_blocks for ins in b.get("insights", []) if ins]) or "No summary provided."
            meta = {
                "insights": insights_joined,
                "metrics_short": ", ".join([b.get("title", "") for b in data_blocks]),
            }
            with open(os.path.join(TMP_DIR, f"{module_name}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)

            st.info("🧩 A copy of this report has been added to the consolidated deck queue.")

        except Exception as e:
            st.error(f"⚠️ PDF generation failed: {e}")
        finally:
            buf.close()