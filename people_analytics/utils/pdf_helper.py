# utils/pdf_helper.py
import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch

# ==========================
# ReportLab-Based PDF Export
# ==========================

def render_pdf_download_button(title: str, dataframe, filename: str):
    """
    Generate a simple ReportLab PDF export for analytics tables.
    
    Args:
        title (str): Section title for the PDF
        dataframe (pd.DataFrame): The table to export
        filename (str): The name of the downloadable file (e.g., "Performance_Report.pdf")
    """

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # --- Header ---
    pdf.setFillColor(colors.HexColor("#1E3A8A"))
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(width / 2, height - 70, title)

    # --- Subtitle ---
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.black)
    pdf.drawCentredString(width / 2, height - 90, "Generated via Streamlit | ReportLab Engine")

    # --- Draw Table ---
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    data = [list(dataframe.columns)] + dataframe.values.tolist()
    table = Table(data, colWidths=[1.8 * inch] * len(dataframe.columns))

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ])
    table.setStyle(style)

    table_width, table_height = table.wrap(0, 0)
    table.drawOn(pdf, 40, height - 130 - table_height)

    # --- Footer ---
    pdf.setFont("Helvetica-Oblique", 8)
    pdf.setFillColor(colors.grey)
    pdf.drawString(40, 40, "Prepared by Amlan Mishra | HR Tech Portfolio | © 2025")

    pdf.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()

    # --- Streamlit Download Button ---
    st.download_button(
        label=f"⬇️ Download {title} (PDF)",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True
    )