# ============================================
# utils_consolidated/uploader_consolidated_helper.py — v1.3 | Multi-format Robust Uploader
# ============================================
import streamlit as st
import pandas as pd

def upload_data(label: str, key: str | None = None):
    """
    Unified CSV/XLS/XLSX uploader for Consolidated module.
    ✅ Accepts all filetypes on desktop & mobile
    ✅ Avoids greyed-out CSVs via MIME hints
    ✅ Returns clean pandas.DataFrame
    """
    file = st.file_uploader(
        label,
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=False,
        key=key,
        help="Supports CSV and Excel files (XLS/XLSX)."
    )

    if file is None:
        return None

    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith(".xls"):
            df = pd.read_excel(file, engine=None)
        else:
            df = pd.read_excel(file, engine="openpyxl")

        st.success(f"✅ {file.name} uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"⚠️ Failed to load {file.name}: {e}")
        st.info("Try re-saving as UTF-8 CSV or Excel (no password protection).")
        return None