# ============================================
# utils_consolidated/chart_consolidated_saver.py
# v5.3 | Cloud-Stable Chart Saver for ReportLab PDFs
# ============================================

import os
import io
import time
import traceback
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------
# Optional Dependencies (graceful fallbacks)
# -----------------------------------------------------------
try:
    import plotly.io as pio
except Exception:
    pio = None

try:
    from PIL import Image
except Exception:
    Image = None

# -----------------------------------------------------------
# ⚙️ Config
# -----------------------------------------------------------
TMP_DIR = os.path.join("/tmp", "consolidated_charts")
os.makedirs(TMP_DIR, exist_ok=True)

PALETTE = px.colors.qualitative.Vivid


# -----------------------------------------------------------
# 🎨 Theme Ensurer
# -----------------------------------------------------------
def _apply_color_theme(fig):
    """Ensure bright visuals (white background, vivid colors)."""
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
            if hasattr(trace, "line"):
                if getattr(trace.line, "color", None) in [None, "#000", "black"]:
                    trace.line.color = color
    except Exception as e:
        st.warning(f"⚠️ Color theme adjustment failed: {e}")


# -----------------------------------------------------------
# 🧩 Fallback to Pillow (white background)
# -----------------------------------------------------------
def _ensure_rgb_white_background(image_bytes: bytes, out_path: str):
    """Ensure all exported charts have white backgrounds (for PDFs)."""
    try:
        if not Image:
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            return out_path

        with Image.open(io.BytesIO(image_bytes)) as im:
            if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                im = im.convert("RGBA")
                bg.paste(im, mask=im.split()[-1])
                bg.save(out_path, "PNG")
            else:
                im.convert("RGB").save(out_path, "PNG")
        return out_path

    except Exception as e:
        st.warning(f"⚠️ PNG post-processing failed: {e}")
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        return out_path


# -----------------------------------------------------------
# 🖼️ Main Export Function
# -----------------------------------------------------------
def save_chart_image(title: str, fig, width: int = 1200, height: int = 700, scale: int = 2):
    """
    Converts a Plotly figure into a static PNG file usable in ReportLab PDFs.
    Includes fallback handling for environments without Kaleido.
    """
    safe_name = title.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(TMP_DIR, f"{safe_name}.png")

    try:
        _apply_color_theme(fig)

        if pio is None:
            raise RuntimeError("Plotly I/O unavailable — Kaleido not installed or broken.")

        image_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        if not image_bytes or len(image_bytes) < 100:
            raise RuntimeError("Empty PNG bytes from Kaleido export.")

        _ensure_rgb_white_background(image_bytes, out_path)

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise IOError(f"PNG export failed for '{title}'")

        return out_path

    except Exception as e:
        st.warning(f"⚠️ Chart export failed for '{title}': {e}")
        st.write(traceback.format_exc())

        # Minimal fallback: save as blank PNG
        if Image:
            fallback = Image.new("RGB", (800, 400), color=(255, 255, 255))
            fallback.save(out_path, "PNG")
            return out_path
        else:
            return None


# -----------------------------------------------------------
# 🔁 Retry Wrapper
# -----------------------------------------------------------
def ensure_chart_saved(title: str, fig, attempts: int = 3, wait: float = 0.3):
    """Retries chart export multiple times — resilient under load."""
    for i in range(attempts):
        path = save_chart_image(title, fig)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        time.sleep(wait * (i + 1))
    return None