# ============================================
# utils/uploader_helper.py — v1.2 | Fully Mobile-Compatible Universal Uploader
# ============================================

import streamlit as st
import pandas as pd

def upload_data(label: str = "Upload your file", help_text: str = None):
    """
    Universal file uploader.
    ✅ Works on mobile (Android/iOS)
    ✅ Accepts any file extension; checks content internally
    ✅ Handles CSV, XLSX, XLS, TXT gracefully
    """
    uploaded_file = st.file_uploader(
        label,
        type=None,  # ← allows all file types; OS won't grey out CSVs
        accept_multiple_files=False,
        help=help_text,
        key=f"upload_{label.replace(' ', '_')}"
    )

    if not uploaded_file:
        return None

    try:
        name = uploaded_file.name.lower()
        if name.endswith((".csv", ".txt")):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            # fallback: try reading as CSV first
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
        return None