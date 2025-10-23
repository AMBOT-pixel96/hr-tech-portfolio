# ============================================
# utils/chart_saver.py — v3.0.2 | Dual-Theme + Sync Save (Final Stable)
# ============================================
import os
import time
import streamlit as st
import pandas as pd

def save_chart_image(title, fig):
    """
    Saves Plotly chart as high-quality PNG inside temp_charts directory.
    ✅ Keeps color fidelity in PDFs.
    ✅ Restores dark theme visuals inside the app.
    ✅ Waits for disk write completion to avoid broken PDFs.
    """
    try:
        TMP_DIR = "temp_charts"
        os.makedirs(TMP_DIR, exist_ok=True)
        # sanitize filename: remove slashes or invalid characters
        safe_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title)
        img_path = os.path.join(TMP_DIR, f"{safe_title.replace(' ', '_')}.png")

        # 🧠 Smart dual-theme handling
        orig_layout = fig.layout.to_plotly_json()  # backup layout before altering

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
        )

        # Save synchronously
        fig.write_image(img_path, width=1200, height=700, scale=2)

        # 🕒 small wait to ensure file flushes completely
        time.sleep(0.25)

        # Restore original layout for Streamlit display
        fig.update_layout(**orig_layout)

        # Validation check
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            raise IOError(f"Chart '{title}' save incomplete or empty file.")

        return img_path

    except Exception as e:
        st.warning(f"⚠️ Could not save chart '{title}': {e}")
        return None


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def ensure_chart_saved(fig, title, saver_func):
    """
    Guarantees chart is saved to disk before PDF generation.
    Retries once if the first attempt failed or incomplete.
    """
    if fig is None:
        return None
    try:
        path = saver_func(title, fig)
        time.sleep(0.25)
        if not path or not os.path.exists(path):
            # retry once
            time.sleep(0.3)
            path = saver_func(title + "_retry", fig)
        return path
    except Exception as e:
        st.warning(f"⚠️ Chart save retry failed for '{title}': {e}")
        return None


def safe_categorical(df, col):
    """
    Ensures categorical columns won't crash when new bins appear
    (e.g., TenureCohort in Attrition).
    Converts to string safely if it's a categorical dtype.
    """
    try:
        if col in df.columns and pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)
        return df
    except Exception:
        # fallback — just return unchanged if something goes wrong
        return df