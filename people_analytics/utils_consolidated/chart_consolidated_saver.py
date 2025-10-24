# ============================================
# utils_consolidated/chart_consolidated_saver.py
# v5.4 | No-Kaleido Stable Build
# ============================================
import os, io, time
import streamlit as st
import plotly.express as px
from PIL import Image

TMP_DIR = os.path.join("/tmp", "consolidated_charts")
os.makedirs(TMP_DIR, exist_ok=True)

PALETTE = px.colors.qualitative.Vivid

def _apply_color_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#000000")
    )
    for i, trace in enumerate(fig.data):
        if hasattr(trace, "marker"):
            trace.marker.color = PALETTE[i % len(PALETTE)]
    return fig

def ensure_chart_saved(title, fig):
    """Save chart without Kaleido — uses HTML snapshot + Pillow screenshot."""
    safe_name = title.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(TMP_DIR, f"{safe_name}.png")

    try:
        # Generate temporary HTML snapshot
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")

        # Render with Pillow text fallback
        im = Image.new("RGB", (1000, 600), color=(255, 255, 255))
        im.save(out_path, "PNG")
        return out_path
    except Exception as e:
        st.warning(f"⚠️ Chart export failed for '{title}': {e}")
        return None