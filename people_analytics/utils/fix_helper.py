# ============================================
# utils/fix_helper.py — v3.0.3 | Final Sync + PNG Validation
# ============================================
import os
import time
import streamlit as st

def ensure_chart_saved(fig, title, saver_func, retries=3, wait=0.4):
    """
    Guarantees chart PNG is created before PDF generation.
    Retries up to `retries` times until file exists and is non-empty.
    """
    if fig is None:
        return None
    try:
        for attempt in range(retries):
            path = saver_func(title, fig)
            if path and os.path.exists(path) and os.path.getsize(path) > 5000:
                # success — file saved and valid
                return path
            time.sleep(wait)  # wait before retry
        st.warning(f"⚠️ Chart image not found after {retries} retries for '{title}'")
        return None
    except Exception as e:
        st.warning(f"⚠️ Chart save failed for '{title}': {e}")
        return None


def safe_categorical(df, col):
    """
    Ensures categorical columns won't crash when new bins appear.
    Converts to string safely if it's a categorical dtype.
    """
    import pandas as pd
    try:
        if col in df.columns and pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)
        return df
    except Exception:
        return df