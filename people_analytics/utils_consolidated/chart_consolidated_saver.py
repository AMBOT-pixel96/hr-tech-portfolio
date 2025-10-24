# ============================================
# utils_consolidated/chart_consolidated_saver.py — v6.4 | Zero-Byte Safe Edition
# ============================================
import os, io, time, traceback
import streamlit as st
import plotly.express as px
from PIL import Image

try:
    import plotly.io as pio
except Exception:
    pio = None

TMP_DIR = os.path.join("/tmp", "consolidated_charts")
os.makedirs(TMP_DIR, exist_ok=True)
PALETTE = px.colors.qualitative.Vivid

# ---------------------------------------------------------
# 🧩 Apply Color Theme
# ---------------------------------------------------------
def _apply_color_theme(fig):
    """Ensure bright visuals for export (white BG + vivid palette)."""
    try:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
        )
        for i, trace in enumerate(fig.data):
            color = PALETTE[i % len(PALETTE)]
            if trace.type == "pie":
                trace.marker.colors = PALETTE[: len(trace.labels)]
                trace.marker.line = dict(width=1, color="#FFFFFF")
            elif hasattr(trace, "marker"):
                trace.marker.color = color
                trace.marker.line = dict(width=0.8, color="#333333")
    except Exception as e:
        st.warning(f"⚠️ Color fidelity patch failed: {e}")

# ---------------------------------------------------------
# 🧩 Save Chart Image (Safe Kaleido Export)
# ---------------------------------------------------------
def save_chart_image(title: str, fig, width=1200, height=700, scale=2):
    try:
        if fig is None or not getattr(fig, "data", []):
            raise ValueError("Empty or invalid figure object.")

        safe_name = title.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(TMP_DIR, f"{safe_name}.png")
        _apply_color_theme(fig)

        if not pio:
            raise RuntimeError("Plotly I/O unavailable (Kaleido missing).")

        image_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        if not image_bytes or len(image_bytes) < 500:
            raise ValueError(f"Invalid or empty image bytes for {title}")

        # Ensure non-transparent RGB output
        with Image.open(io.BytesIO(image_bytes)) as im:
            bg = Image.new("RGB", im.size, (255, 255, 255))
            if im.mode == "RGBA":
                bg.paste(im, mask=im.split()[-1])
            else:
                bg.paste(im)
            bg.save(out_path, "PNG")

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 500:
            raise IOError(f"PNG not written correctly for {title}")

        return out_path

    except Exception as e:
        st.error(f"⚠️ Chart export failed for '{title}': {e}")
        st.code(traceback.format_exc())
        return None

# ---------------------------------------------------------
# 🧩 Retry Wrapper
# ---------------------------------------------------------
def ensure_chart_saved(title: str, fig, attempts: int = 3, wait: float = 0.3):
    """Retry export multiple times, abort if no valid file."""
    for i in range(attempts):
        path = save_chart_image(title, fig)
        if path and os.path.exists(path) and os.path.getsize(path) > 500:
            return path
        time.sleep(wait * (i + 1))
    st.warning(f"⚠️ Chart '{title}' could not be saved after {attempts} attempts.")
    return None