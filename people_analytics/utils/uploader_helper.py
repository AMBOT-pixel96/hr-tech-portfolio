# ============================================
# utils/uploader_helper.py — v1.0 | Universal Upload Fix
# ============================================

import streamlit as st
import pandas as pd

# --- All supported MIME types for CSV/Excel ---
_SUPPORTED_TYPES = [
    "text/csv",
    "text/plain",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
]

def upload_data(label: str = "Upload your file", help_text: str = None):
    """
    Universal file uploader to fix Streamlit CSV grey-out issue.

    Returns:
        pd.DataFrame or None
    """
    uploaded_file = st.file_uploader(
        label,
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        help=help_text,
        key=f"upload_{label.replace(' ', '_')}"
    )

    if not uploaded_file:
        return None

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
        return None