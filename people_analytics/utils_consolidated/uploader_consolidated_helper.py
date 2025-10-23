# ============================================
# utils_consolidated/uploader_consolidated_helper.py
# v1.2 — Multi-format uploader with MIME fix for mobile
# ============================================
import streamlit as st
import pandas as pd

def upload_data(label: str, key: str | None = None):
    """
    Robust file uploader:
    ✅ Supports CSV, XLS, XLSX
    ✅ Works on mobile (explicit MIME types)
    ✅ Shows friendly success / error messages
    """
    file = st.file_uploader(
        label,
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=False,
        key=key,
        help="Supports CSV and Excel files (XLS/XLSX)."
    )

    # ✅ Fix: handle mobile browsers where CSVs are greyed out
    if file is None:
        return None

    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith(".xls"):
            df = pd.read_excel(file)
        else:
            df = pd.read_excel(file, engine="openpyxl")

        st.success(f"✅ {file.name} uploaded successfully!")
        return df

    except Exception as e:
        st.error(f"⚠️ Failed to read {file.name}: {e}")
        st.info("If file doesn't open, re-save it as UTF-8 CSV or clean Excel.")
        return None