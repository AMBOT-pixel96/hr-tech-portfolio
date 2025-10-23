# ============================================
# utils_consolidated/uploader_consolidated_helper.py
# v1.0 — Robust multi-format uploader for Consolidated module
# ============================================
import streamlit as st
import pandas as pd

def upload_data(label: str, key: str | None = None):
    """
    Unified CSV / XLS / XLSX uploader for the consolidated workflow.
    - Accepts csv, xls, xlsx
    - Returns a pandas.DataFrame or None on cancel/error
    - Shows friendly success / error messages in-app
    """
    file = st.file_uploader(label, type=["csv", "xls", "xlsx"], key=key)
    if file is None:
        return None

    try:
        # prefer pandas engine autodetection, but force openpyxl for xlsx when needed
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith(".xls"):
            # xlrd may be unavailable; pandas can often handle .xls — try fallback to engine if needed
            df = pd.read_excel(file)
        else:  # .xlsx
            df = pd.read_excel(file, engine="openpyxl")
        st.success(f"✅ {file.name} uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"⚠️ Failed to read {file.name}: {e}")
        # helpful hint for users
        st.info("If the file is large, try saving as CSV. For .xlsx ensure it isn't password protected.")
        return None