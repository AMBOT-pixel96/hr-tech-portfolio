# ============================================
# utils/chart_saver.py — v3.1.2 | JSON-safe, Sync Save, Dual Theme (FINAL)
# ============================================
import os
import time
import json
import streamlit as st
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder


def save_chart_image(title, fig):
    """
    Saves Plotly chart as a high-quality PNG inside temp_charts directory.
    ✅ Full-color fidelity in PDFs (plotly_white template)
    ✅ Restores dark/light theme for app preview
    ✅ Handles ndarray serialization safely
    ✅ Waits until disk write is complete before returning
    """
    try:
        TMP_DIR = "temp_charts"
        os.makedirs(TMP_DIR, exist_ok=True)
        img_path = os.path.join(TMP_DIR, f"{title.replace(' ', '_')}.png")

        # Backup original layout
        fig_json = fig.to_plotly_json()
        orig_layout = fig_json.get("layout", {}).copy()

        # --- White theme overrides for PDF export
        layout_updates = {
            "template": "plotly_white",
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#FFFFFF",
            "font": {"color": "#000000"},
        }
        fig_json["layout"] = {**orig_layout, **layout_updates}

        # --- Use PlotlyJSONEncoder to handle ndarrays safely
        fig_for_pdf = pio.from_json(json.dumps(fig_json, cls=PlotlyJSONEncoder))

        # --- Write image to disk (sync + retry)
        for attempt in range(3):
            try:
                fig_for_pdf.write_image(img_path, width=1200, height=700, scale=2)
                time.sleep(0.4)
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    break
            except Exception as write_err:
                time.sleep(0.6)
                if attempt == 2:
                    raise write_err

        # --- Verify file write success
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            raise IOError(f"Chart image not found after retries for '{title}'")

        # --- Restore original layout for app display
        fig.update_layout(**orig_layout)

        return img_path

    except Exception as e:
        st.warning(f"⚠️ Could not save chart '{title}': {e}")
        return None


# ============================================
# ⏳ Support Functions
# ============================================
import pandas as pd


def ensure_chart_saved(fig, title):
    """
    Guarantees chart is saved before PDF generation.
    Adds safety delay to ensure write completion.
    """
    try:
        path = save_chart_image(title, fig)
        time.sleep(0.3)
        if path and os.path.exists(path):
            return path
        else:
            st.warning(f"⚠️ Chart image for '{title}' was not saved correctly.")
            return None
    except Exception as e:
        st.warning(f"⚠️ Failed to ensure chart save for '{title}': {e}")
        return None


def safe_categorical(df, col):
    """
    Converts pandas Categorical columns to string to prevent category assignment errors.
    Example: prevents 'Cannot setitem on a Categorical with a new category' crash.
    """
    if col in df.columns and pd.api.types.is_categorical_dtype(df[col]):
        df[col] = df[col].astype(str)
    return df