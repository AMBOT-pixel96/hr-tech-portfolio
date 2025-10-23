# ============================================
# utils_consolidated/uploader_helper.py — v5.1 | Safe Multi-Format Upload
# ============================================
import streamlit as st
import pandas as pd

def upload_data(label, key=None):
    """
    Robust CSV/XLSX uploader for all modules.
    Handles gray-out issues & ensures consistent parsing.
    """
    file = st.file_uploader(label, type=["csv", "xlsx", "xls"], key=key)
    if not file:
        return None

    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine="openpyxl")
        st.success(f"✅ {file.name} uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"⚠️ Failed to read {file.name}: {e}")
        return None