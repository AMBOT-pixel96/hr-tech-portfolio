# ============================================
# utils_consolidated/uploader_consolidated_helper.py
# v1.1 — Robust multi-format uploader (mobile & CSV friendly)
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
    file = st.file_uploader(
        label,
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=False,
        key=key
    )

    st.caption("📁 Supported formats: CSV, XLS, XLSX — ensure files aren't password protected or larger than 200MB.")

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
        st.info("Tip: Try saving as CSV if this persists or ensure Excel file isn’t encrypted.")
        return None