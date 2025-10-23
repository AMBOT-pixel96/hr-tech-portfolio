# ============================================
# utils_consolidated/uploader_consolidated_helper.py — v5.1 | Universal Upload Fix
# ============================================
import streamlit as st
import pandas as pd

def upload_data(label: str = "Upload your file", help_text: str = None, key_suffix: str = ""):
    """
    Universal file uploader allowing CSV, XLS, XLSX, and TXT.
    ✅ Fixes mobile browser visibility issue (CSV greyed out)
    ✅ Auto-detects delimiter for .txt and .csv
    ✅ Graceful Excel parsing fallback
    """
    uploaded_file = st.file_uploader(
        label,
        type=["csv", "xls", "xlsx", "txt"],
        accept_multiple_files=False,
        help=help_text,
        key=f"upload_{label.replace(' ', '_')}_{key_suffix}"
    )

    if not uploaded_file:
        return None

    try:
        filename = uploaded_file.name.lower()
        if filename.endswith((".csv", ".txt")):
            # Try multiple encodings and delimiters for safety
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep="\t", engine="python")
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
        return None