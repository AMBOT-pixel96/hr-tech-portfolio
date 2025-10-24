# utils_consolidated/uploader_consolidated_helper.py
# v1.3 — uploader that is mobile-friendly and explicit about formats
import streamlit as st
import pandas as pd

def upload_data(label: str, key: str = None):
    """
    Robust file uploader:
      - Accepts CSV, XLS, XLSX
      - Returns DataFrame or None
      - Explicit caption helps mobile file chooser
    """
    file = st.file_uploader(
        label,
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=False,
        key=key,
        help="Supported: CSV, XLS, XLSX — max ~200MB per file (Streamlit limits dependent)."
    )

    # If user hasn't picked a file yet, return None (UI shows uploader)
    if file is None:
        return None

    # Try to read file robustly
    try:
        name = file.name.lower()
        # read in-memory
        if name.endswith(".csv"):
            # explicit encoding if user environment needs it
            try:
                df = pd.read_csv(file)
            except Exception:
                # try with utf-8-sig fallback
                df = pd.read_csv(file, encoding="utf-8-sig")
        elif name.endswith(".xls"):
            df = pd.read_excel(file)
        else:  # .xlsx
            df = pd.read_excel(file, engine="openpyxl")
        st.success(f"✅ {file.name} uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"⚠️ Failed to read {getattr(file, 'name', 'file')}: {e}")
        st.info("If the file doesn't load: try saving as CSV (UTF-8) or ensure .xlsx isn't password-protected.")
        return None