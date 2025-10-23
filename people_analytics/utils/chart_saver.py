# ============================================
# utils/chart_saver.py — v3.1 | Synchronous Kaleido Lock (Final)
# ============================================
import os
import time
import streamlit as st
import pandas as pd
import plotly.io as pio

def save_chart_image(title, fig):
    """
    Fully synchronous chart saver.
    ✅ Forces Kaleido to render before returning.
    ✅ Keeps color fidelity in PDFs.
    ✅ Works on Streamlit Cloud and local.
    """
    TMP_DIR = "temp_charts"
    os.makedirs(TMP_DIR, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title)
    img_path = os.path.join(TMP_DIR, f"{safe_title}.png")

    try:
        # temporarily force light theme for PDFs
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
        )

        # --- explicit Kaleido engine call ---
        pio.write_image(fig, img_path, format="png", width=1200, height=700, scale=2, engine="kaleido")

        # block until file exists and non-empty
        for _ in range(10):
            if os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
                break
            time.sleep(0.1)

        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            raise IOError("PNG not generated or empty.")

        return img_path

    except Exception as e:
        st.warning(f"⚠️ Could not save chart '{title}': {e}")
        return None


def ensure_chart_saved(fig, title, saver_func=save_chart_image):
    """Wrapper that ensures file truly exists before PDF build."""
    if fig is None:
        return None
    path = saver_func(title, fig)
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    # one retry
    time.sleep(0.5)
    return saver_func(title + "_retry", fig)


def safe_categorical(df, col):
    """Safely converts categorical columns to string before manipulation."""
    try:
        if col in df.columns and pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)
    except Exception:
        pass
    return df