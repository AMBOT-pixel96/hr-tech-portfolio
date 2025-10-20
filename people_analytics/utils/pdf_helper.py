# utils/pdf_helper.py
"""
Streamlit helper for showing PDF download button that calls pdf_auto_exporter.export_module_report
"""

import streamlit as st
from .pdf_auto_exporter import export_module_report

def render_pdf_download_button(report_title: str, module_name: str, data_blocks: list, filename_prefix: str = None):
    """
    Calls export_module_report and renders a Streamlit download button with the produced PDF.
    - report_title: displayed inside PDF (cover)
    - module_name: name shown on PDF cover
    - data_blocks: same structure as pdf_auto_exporter expects
    - filename_prefix: used for download filename
    """
    try:
        pdf_bytes = export_module_report(report_title=report_title, module_name=module_name, data_blocks=data_blocks, filename_prefix=filename_prefix)
        fname = f"{(filename_prefix or module_name).replace(' ','_')}_Executive_Report.pdf"
        st.download_button(
            label="⬇️ Download Executive PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Could not generate PDF: {e}")