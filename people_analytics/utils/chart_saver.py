# ============================================
# utils/chart_saver.py — v3.1.0 | Dual-Theme + Sync Save (stable)
# ============================================
"""
Robust chart saver for Plotly figures used in PDF exports.

Usage:
    from utils.chart_saver import save_chart_image, ensure_chart_saved, safe_categorical

Notes:
- Produces PNGs with a white background and black text (good for PDFs).
- Works with kaleido (plotly engine). Waits and fsyncs to ensure file is fully written.
- Operates on a deep-copy of the figure to avoid mutating the figure shown in the Streamlit app.
"""
import os
import time
import json
import tempfile
import hashlib
from pathlib import Path

import pandas as pd
import plotly.io as pio
from plotly.graph_objs import Figure
import streamlit as st

TMP_DIR = "temp_charts"
os.makedirs(TMP_DIR, exist_ok=True)


def _sanitize_filename(title: str) -> str:
    """Create safe filename from title (keeps it short & unique)."""
    if not title:
        title = "chart"
    # basic sanitization
    safe = "".join(c if c.isalnum() or c in "-_. " else "_" for c in title).strip()
    # shorten and append hash for uniqueness
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    name = f"{safe[:60].strip().replace(' ', '_')}_{h}.png"
    return name


def _write_bytes_atomic(path: str, data: bytes, attempts: int = 3, wait: float = 0.12):
    """
    Write bytes to file atomically and fsync to ensure disk flush.
    Retries a few times if the file is incomplete.
    """
    for attempt in range(attempts):
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            # final check
            if os.path.exists(path) and os.path.getsize(path) > 64:
                return True
        except Exception:
            # best-effort cleanup
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
        time.sleep(wait * (attempt + 1))
    return False


def save_chart_image(title: str, fig: Figure, width: int = 1200, height: int = 700, scale: int = 2) -> str | None:
    """
    Save a Plotly figure to PNG suitable for embedding into PDFs.

    - Does not mutate the original figure shown in the Streamlit UI.
    - Forces a light template for the exported image so colors remain visible in PDFs.
    - Ensures the saved file is fully written (fsync + validation).

    Returns the path to the saved PNG or None on failure.
    """
    try:
        if fig is None:
            raise ValueError("No figure provided")

        # create copy of figure via JSON roundtrip to avoid mutating original
        fig_json = fig.to_plotly_json()
        # Apply light-mode overrides for PDF export on the copied JSON
        layout = fig_json.get("layout", {})
        # Set white background and black text for PDF-friendly output
        layout_updates = {
            "template": "plotly_white",
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#FFFFFF",
            "font": {"color": "#000000"},
        }
        # Merge without destroying other layout keys
        layout.update(layout_updates)
        fig_json["layout"] = layout

        # Convert back to a figure object for writing
        fig_for_pdf = pio.from_json(json.dumps(fig_json))

        filename = _sanitize_filename(title)
        img_path = os.path.join(TMP_DIR, filename)

        # Use plotly.io.to_image to get bytes first (so we can atomic-write)
        img_bytes = pio.to_image(fig_for_pdf, format="png", width=width, height=height, scale=scale, validate=True)

        ok = _write_bytes_atomic(img_path, img_bytes)
        if not ok:
            raise IOError("Failed to write image atomically")

        # final safety pause & verify
        timeout = 2.0
        start = time.time()
        while (not os.path.exists(img_path) or os.path.getsize(img_path) == 0) and time.time() - start < timeout:
            time.sleep(0.05)

        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            raise IOError("File save incomplete or empty after final verification")

        return img_path

    except Exception as e:
        # Streamlit warning is useful in dev — keep message concise
        st.warning(f"⚠️ Could not save chart '{title}': {str(e)}")
        return None


def ensure_chart_saved(fig: Figure, title: str, saver_func=save_chart_image, retries: int = 2) -> str | None:
    """
    Wrapper that guarantees the saver returns a usable path, with a retry fallback.
    Returns the saved path or None on failure.
    """
    last = None
    for attempt in range(retries + 1):
        path = saver_func(title, fig)
        if path and os.path.exists(path) and os.path.getsize(path) > 64:
            return path
        last = path
        time.sleep(0.15 * (attempt + 1))
    # final attempt result (could be None)
    return last


def safe_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Ensure categorical dtypes don't cause errors during PDF rendering or when
    converting dataframes to string. If the column is categorical, convert to str.
    Returns the dataframe (modified copy).
    """
    df2 = df.copy()
    if col in df2.columns:
        try:
            if pd.api.types.is_categorical_dtype(df2[col]):
                df2[col] = df2[col].astype(str)
        except Exception:
            # be defensive: coerce to string anyway
            df2[col] = df2[col].astype(str)
    return df2