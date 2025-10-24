# ============================================
# utils_consolidated/pdf_merger.py — v1.0 | Cloud-Safe Consolidated Merger
# ============================================
import os
import streamlit as st
from PyPDF2 import PdfMerger

TMP_DIR = os.path.join("/tmp", "consolidated_pdfs")
os.makedirs(TMP_DIR, exist_ok=True)

def list_existing_pdfs():
    """List all module PDFs already saved in /tmp."""
    return [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]

def save_uploaded_pdf(file, module_name):
    """Optional manual upload for testing."""
    out_path = os.path.join(TMP_DIR, f"{module_name}.pdf")
    with open(out_path, "wb") as f:
        f.write(file.getbuffer())
    return out_path

def merge_pdfs(output_name="Consolidated_Leadership_Deck.pdf"):
    """Merges all PDFs from TMP_DIR into one master file."""
    merger = PdfMerger()
    pdfs = sorted(list_existing_pdfs())

    if not pdfs:
        st.warning("⚠️ No module PDFs found to merge.")
        return None

    try:
        for pdf in pdfs:
            merger.append(os.path.join(TMP_DIR, pdf))
        out_path = os.path.join(TMP_DIR, output_name)
        merger.write(out_path)
        merger.close()
        return out_path
    except Exception as e:
        st.error(f"❌ PDF merge failed: {e}")
        return None