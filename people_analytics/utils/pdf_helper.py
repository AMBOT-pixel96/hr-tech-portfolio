# ============================================
# utils/pdf_helper.py — v1.0 | WeasyPrint PDF Generator
# ============================================

import streamlit as st
from weasyprint import HTML, CSS
from datetime import datetime
import os

# Ensure export directory exists
EXPORT_DIR = os.path.join(os.getcwd(), "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

def generate_pdf_report(title: str, html_content: str, filename_prefix: str = "Report"):
    """
    Generates a fully-styled PDF report from an HTML block using WeasyPrint.
    
    Args:
        title (str): Report title (e.g., "Performance Analytics Report")
        html_content (str): The full HTML of the report (charts excluded, but can include images)
        filename_prefix (str): Base filename for the exported PDF
        
    Returns:
        str: Path to the generated PDF
    """

    # --- Define output file ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.pdf"
    output_path = os.path.join(EXPORT_DIR, filename)

    # --- PDF CSS Theme (Brand + Matte) ---
    css = CSS(string="""
        @page {
            size: A4;
            margin: 1.2cm;
        }
        body {
            font-family: 'Open Sans', sans-serif;
            color: #111827;
            background: #FAFAFA;
            line-height: 1.5;
            font-size: 11pt;
        }
        h1, h2, h3 {
            color: #1E3A8A;
            font-weight: 700;
        }
        h1 {
            font-size: 18pt;
            border-bottom: 2px solid #1E3A8A;
            padding-bottom: 4px;
        }
        h2 {
            font-size: 14pt;
            margin-top: 18px;
        }
        .summary {
            background: #EFF6FF;
            border-left: 5px solid #3B82F6;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .footer {
            text-align: center;
            font-size: 9pt;
            color: #6B7280;
            margin-top: 40px;
            border-top: 1px solid #E5E7EB;
            padding-top: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            border: 1px solid #D1D5DB;
            padding: 6px 8px;
            font-size: 10pt;
        }
        th {
            background: #1E3A8A;
            color: white;
        }
        tr:nth-child(even) {
            background: #F3F4F6;
        }
    """)

    # --- Build final HTML ---
    html_template = f"""
    <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
        </head>
        <body>
            <h1>{title}</h1>
            {html_content}
            <div class="footer">
                Prepared with ❤️ by Amlan Mishra | © 2025 HR Tech Portfolio
            </div>
        </body>
    </html>
    """

    # --- Generate PDF ---
    HTML(string=html_template).write_pdf(output_path, stylesheets=[css])
    return output_path


def render_pdf_download_button(title: str, html_content: str, filename_prefix: str = "Report"):
    """
    Generates a PDF on-demand and shows a Streamlit download button.
    """
    try:
        pdf_path = generate_pdf_report(title, html_content, filename_prefix)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label=f"📄 Download {title} (PDF)",
            data=pdf_bytes,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            use_container_width=True
        )
        st.success("✅ PDF generated successfully!")
    except Exception as e:
        st.error(f"⚠️ PDF generation failed: {e}")