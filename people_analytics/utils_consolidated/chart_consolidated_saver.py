# utils_consolidated/chart_consolidated_saver.py
"""
Color-safe Kaleido exporter for consolidated PDF engine.
Provides:
 - save_chart_image(title, fig) -> path or None
 - ensure_chart_saved(title, fig) -> path or None (retries)
"""

import os
import io
import time
import traceback
import plotly.express as px
import streamlit as st

try:
    import plotly.io as pio
except Exception:
    pio = None

try:
    from PIL import Image
except Exception:
    Image = None

from .constants import EXPORT

TMP_DIR = os.path.join("/tmp", "consolidated_temp_charts")
os.makedirs(TMP_DIR, exist_ok=True)

PALETTE = px.colors.qualitative.Vivid

# small helpers
def _ensure_png_has_white_bg_from_bytes(b: bytes, out_path: str):
    """Use Pillow to composite transparent PNG over white, else raw write fallback."""
    if Image is None:
        with open(out_path, "wb") as f:
            f.write(b)
        time.sleep(0.08)
        return out_path

    try:
        with Image.open(io.BytesIO(b)) as im:
            if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                im = im.convert("RGBA")
                bg.paste(im, mask=im.split()[-1])
                bg.save(out_path, "PNG")
            else:
                im.convert("RGB").save(out_path, "PNG")
        time.sleep(0.04)
        return out_path
    except Exception as e:
        st.warning(f"⚠️ PNG post-process failed: {e}")
        try:
            with open(out_path, "wb") as f:
                f.write(b)
            return out_path
        except Exception:
            return None

def _apply_color_fix(fig):
    """Apply conservative palette changes to avoid kaleido greyscale artifacts."""
    try:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
        )
        for i, trace in enumerate(fig.data):
            color = PALETTE[i % len(PALETTE)]
            # pies require 'marker.colors' (plural)
            if getattr(trace, "type", None) == "pie":
                trace.marker.colors = PALETTE[: max(1, len(getattr(trace, "labels", [])))]
                trace.marker.line = dict(width=1, color="#FFFFFF")
                continue
            # fallback for many trace types
            if hasattr(trace, "marker"):
                # marker.color might be array or scalar
                try:
                    trace.marker.color = color
                except Exception:
                    # if marker.color expects array, don't force
                    pass
                trace.marker.line = dict(width=0.6, color="#333333")
            if hasattr(trace, "line"):
                if getattr(trace.line, "color", None) in (None, "#000", "black"):
                    trace.line.color = color
    except Exception as e:
        st.warning(f"⚠️ color fix failed: {e}")

def save_chart_image(title: str, fig, filename_safe: str = None, width: int = None, height: int = None, scale: int = None):
    """
    Export fig -> PNG file and return path. Uses in-memory export via Kaleido.
    """
    try:
        if pio is None:
            raise RuntimeError("plotly.io not available (kaleido missing)")

        safe_name = (filename_safe or title).replace(" ", "_").replace("/", "_")
        out_path = os.path.join(TMP_DIR, f"{safe_name}.png")

        # apply fixes
        _apply_color_fix(fig)

        width = width or EXPORT.get("width", 1200)
        height = height or EXPORT.get("height", 700)
        scale = scale or EXPORT.get("scale", 2)

        # produce in-memory PNG bytes
        image_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        if not image_bytes or len(image_bytes) < 100:
            raise RuntimeError("Kaleido returned empty bytes")

        # composite to white bg and save
        saved = _ensure_png_has_white_bg_from_bytes(image_bytes, out_path)
        if not saved or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise IOError("PNG save failed")

        # quick sanity check
        if os.path.getsize(out_path) < 200:
            st.warning(f"⚠️ tiny png produced for '{title}' ({os.path.getsize(out_path)} bytes)")

        return out_path

    except Exception as e:
        st.warning(f"⚠️ Chart save failed for '{title}': {e}")
        st.debug(traceback.format_exc())
        return None

def ensure_chart_saved(title: str, fig, attempts: int = 3, wait: float = 0.25):
    """Retry wrapper for save_chart_image."""
    last = None
    for i in range(attempts):
        path = save_chart_image(title, fig)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        last = path
        time.sleep(wait * (i + 1))
    return None