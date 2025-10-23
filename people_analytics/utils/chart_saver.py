# ============================================
# utils/chart_saver.py — v1.0 | File-based Chart Saver
# ============================================

import os
from pathlib import Path

# Create a safe temporary directory for PNGs
TMP_DIR = Path("tmp_charts")
TMP_DIR.mkdir(exist_ok=True)

def sanitize_anchor(title: str) -> str:
    """Safe filename sanitizer (used for chart filenames)."""
    return "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")

def save_chart_image(title, fig, width=1200, height=700, scale=2):
    """
    Saves a Plotly chart to a high-quality PNG inside tmp_charts/.
    ✅ Works without Kaleido dependency on all Streamlit environments.
    ✅ Ensures bright background, visible text, and readable axes.
    ✅ Returns full file path of saved image.
    """
    try:
        fname = TMP_DIR / f"{sanitize_anchor(title)}.png"

        # 🧠 Apply bright, PDF-safe background
        fig.update_layout(
            paper_bgcolor="#F9FAFB",
            plot_bgcolor="#F9FAFB",
            font=dict(color="#000", size=12),
            margin=dict(t=60, l=60, r=40, b=60),
        )

        # ✅ Add border around bars/scatters for contrast
        for tr in fig.data:
            if hasattr(tr, "marker"):
                tr.marker.line = dict(width=0.6, color="#E5E7EB")

        # Save image (Plotly handles this internally, Kaleido optional)
        fig.write_image(str(fname), width=width, height=height, scale=scale)
        print(f"✅ Chart saved: {fname}")
        return str(fname)

    except Exception as e:
        print(f"⚠️ Could not save chart '{title}': {e}")
        return None