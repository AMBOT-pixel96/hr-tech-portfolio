# ============================================
# utils/uploader_helper.py — v1.1 | Universal Upload (CSV/XLSX/TXT)
# ============================================

import streamlit as st
import pandas as pd

def upload_data(label: str = "Upload your file", help_text: str = None):
    """
    Universal file uploader allowing CSV, XLS, XLSX, and TXT files.
    ✅ Fixes mobile issue where only recent/xlsx files were visible.
    ✅ Gracefully handles CSV or Excel load errors.
    """
    uploaded_file = st.file_uploader(
        label,
        type=["csv", "xls", "xlsx", "txt"],   # all supported formats
        accept_multiple_files=False,
        help=help_text,
        key=f"upload_{label.replace(' ', '_')}"
    )

    if not uploaded_file:
        return None

    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv") or filename.endswith(".txt"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
        return None