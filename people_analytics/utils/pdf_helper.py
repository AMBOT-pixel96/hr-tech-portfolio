# ============================================
# utils/pdf_helper.py — v3.3 | Auto-detect Path/Figure (FINAL)
# ============================================
import os
import io
import time
import streamlit as st
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from utils.chart_saver import save_chart_image
import plotly.io as pio


def render_pdf_download_button(report_title, module_name, data_blocks, file_prefix):
    """
    Builds and downloads a polished multi-page executive PDF.
    ✅ Accepts both fig objects and file paths (auto-detect)
    ✅ Handles async chart saves and waits for ready files
    ✅ Preserves full-color fidelity (no black graphs)
    """
    if not data_blocks:
        st.warning("⚠️ No data blocks found for this module.")
        return

    if st.button(f"🧾 Generate {module_name} Executive PDF", use_container_width=True):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=28, bottomMargin=28
        )

        styles = getSampleStyleSheet()
        body = ParagraphStyle(
            "body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
        )

        story = []

        # === Cover Page ===
        story.append(Spacer(1, 100))
        story.append(Paragraph(
            f"<para align=center><font size=20 color='#1E3A8A'><b>{report_title}</b></font></para>", body
        ))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"<para align=center><font size=12>{module_name} Module</font></para>", body
        ))
        story.append(Spacer(1, 40))
        story.append(Paragraph(
            f"<para align=center><font size=10>Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}</font></para>", body
        ))
        story.append(PageBreak())

        # === Table of Contents ===
        toc_data = [["#", "Section", "Description", "Page"]]
        for i, block in enumerate(data_blocks, 1):
            toc_data.append([i, block.get("title", ""), block.get("desc", ""), i + 1])
        toc = Table(toc_data, colWidths=[20, 120, 220, 30])
        toc.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(Paragraph("<b>Table of Contents</b>", styles["Heading2"]))
        story.append(Spacer(1, 8))
        story.append(toc)
        story.append(PageBreak())

        # === Each section ===
        summary_data = [["Section", "Key Insights"]]

        for i, block in enumerate(data_blocks, 1):
            title = block.get("title", f"Section {i}")
            desc = block.get("desc", "")
            df = block.get("df", None)
            fig = block.get("fig", None)
            insights = block.get("insights", [])

            story.append(Paragraph(f"<b>{i}. {title}</b>", styles["Heading2"]))
            story.append(Paragraph(desc, body))
            story.append(Spacer(1, 6))

            # Render table
            if df is not None and not df.empty:
                df = df.round(2).astype(str)
                data = [list(df.columns)] + df.values.tolist()
                t = Table(data, repeatRows=1)
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]))
                story.append(t)
                story.append(Spacer(1, 6))

            # Render figure (auto-detect path or object)
            img_path = None
            if fig is not None:
                try:
                    if isinstance(fig, str) and os.path.exists(fig):
                        img_path = fig
                    else:
                        img_path = save_chart_image(title, fig)
                    # 🕒 Wait briefly to ensure file is ready
                    for _ in range(3):
                        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                            break
                        time.sleep(0.3)
                    if img_path and os.path.exists(img_path):
                        story.append(RLImage(img_path, width=160, height=90))
                        story.append(Spacer(1, 8))
                    else:
                        story.append(Paragraph("⚠️ No chart available for this section.", body))
                except Exception as e:
                    story.append(Paragraph(f"⚠️ Could not render chart: {e}", body))
            else:
                story.append(Paragraph("⚠️ No chart available for this section.", body))

            # Insights
            if insights:
                insights_text = " • ".join(insights)
                story.append(Paragraph(f"<font color='#2563EB'><i>{insights_text}</i></font>", body))
            summary_data.append([title, insights_text if insights else ""])
            story.append(PageBreak())

        # === Executive Summary ===
        story.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
        summary_table = Table(summary_data, colWidths=[150, 300])
        summary_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(summary_table)

        # === Build PDF ===
        doc.build(story)

        st.success("✅ Executive PDF generated successfully.")
        st.download_button(
            "⬇️ Download Report",
            buf.getvalue(),
            file_name=f"{file_prefix}_Executive_Report.pdf",
            mime="application/pdf",
        )