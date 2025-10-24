# ============================================
# utils_consolidated/uploader_consolidated_helper.py
# v1.3 — Multi-format uploader with safe read & mobile fixes
# ============================================
import streamlit as st
import pandas as pd
import io

def _read_csv_bytes(file_bytes: bytes):
    """
    Try a robust CSV read with common fallbacks:
      - utf-8
      - latin1
      - engine='python' fallback
    """
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except Exception:
            continue
    # python engine as last resort
    try:
        return pd.read_csv(io.BytesIO(file_bytes), engine="python", errors="replace")
    except Exception:
        raise

def upload_data(label: str, key: str | None = None):
    """
    Robust file uploader:
      - Accepts csv, xls, xlsx
      - Works on mobile (explicit MIME types not exposed to Streamlit, but this provides better UX)
      - Returns pd.DataFrame or None on cancel/error
    """
    file = st.file_uploader(
        label,
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=False,
        key=key,
        help="Supports CSV and Excel files (XLS/XLSX).",
    )

    if file is None:
        return None

    try:
        name = file.name.lower()
        # streamlit provides a file-like object; sometimes using .read() is more robust across platforms
        raw = file.read()

        if name.endswith(".csv"):
            # prefer robust byte-based read (handles mobile/safari quirks)
            df = _read_csv_bytes(raw)
        elif name.endswith(".xls"):
            # pandas will pick appropriate engine for .xls
            df = pd.read_excel(io.BytesIO(raw))
        else:  # .xlsx
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")

        st.success(f"✅ {file.name} uploaded successfully!")
        return df

    except Exception as e:
        st.error(f"⚠️ Failed to read {getattr(file, 'name', 'uploaded file')}: {e}")
        st.info("Tip: If the file is large or oddly encoded, try saving as UTF-8 CSV or a clean XLSX.")
        return None