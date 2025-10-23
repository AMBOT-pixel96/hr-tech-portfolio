# ============================================
# utils/chart_saver.py — v4.4 | ColorSafe Kaleido Export (Final Stable)
# ============================================
import os, io, time, traceback
import streamlit as st
import plotly.express as px

try:
    import plotly.io as pio
except Exception:
    pio = None

try:
    from PIL import Image
except Exception:
    Image = None


# ======================================================
# ⚙️ Configuration
# ======================================================
TMP_DIR = os.path.join("/tmp", "temp_charts")
os.makedirs(TMP_DIR, exist_ok=True)

# Bright, accessible palette (consistent with Streamlit light mode)
PALETTE = px.colors.qualitative.Vivid


# ======================================================
# 🧩 Helpers
# ======================================================
def _apply_color_fidelity_fix(fig):
    """Ensures vivid categorical colors + white background for all chart types."""
    try:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
        )

        for i, trace in enumerate(fig.data):
            color = PALETTE[i % len(PALETTE)]

            # Handle pies first (they break easily)
            if trace.type == "pie":
                trace.marker.colors = PALETTE[: len(trace.labels)]
                trace.marker.line = dict(width=1, color="#FFFFFF")
                continue  # ✅ skip rest to avoid invalid props

            # Handle bars, boxes, scatters, and lines
            if hasattr(trace, "marker"):
                trace.marker.color = color
                trace.marker.line = dict(width=0.8, color="#333333")

            # Line traces
            if hasattr(trace, "line"):
                if getattr(trace.line, "color", None) in [None, "#000", "black"]:
                    trace.line.color = color

    except Exception as e:
        st.warning(f"⚠️ Color fidelity patch failed: {e}")

def _ensure_png_has_white_bg_from_bytes(b: bytes, out_path: str):
    """Composites transparent PNGs over white background (using Pillow)."""
    if Image is None:
        with open(out_path, "wb") as f:
            f.write(b)
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
        return out_path
    except Exception as e:
        st.warning(f"⚠️ PNG post-process failed: {e}")
        with open(out_path, "wb") as f:
            f.write(b)
        return out_path


# ======================================================
# 🖼️ Main Export Function
# ======================================================
def save_chart_image(title: str, fig, filename_safe: str = None, width: int = 1200, height: int = 700, scale: int = 2):
    """Exports Plotly figure as vivid PNG via Kaleido with guaranteed color & contrast."""
    try:
        safe_name = (filename_safe or title).replace(" ", "_").replace("/", "_")
        out_path = os.path.join(TMP_DIR, f"{safe_name}.png")

        if pio is None:
            raise RuntimeError("Plotly I/O not available for image export.")

        # Apply color corrections before exporting
        _apply_color_fidelity_fix(fig)

        # Export via Kaleido to in-memory bytes
        image_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        if not image_bytes or len(image_bytes) < 100:
            raise RuntimeError("Kaleido produced empty or invalid PNG bytes.")

        # Ensure proper RGB composite
        _ensure_png_has_white_bg_from_bytes(image_bytes, out_path)

        # Sanity check
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise IOError(f"File write failed for {out_path}")

        return out_path

    except Exception as e:
        st.error(f"⚠️ Chart save failed for '{title}': {e}")
        st.write(traceback.format_exc())
        return None


# ======================================================
# 🔁 Retry Wrapper
# ======================================================
def ensure_chart_saved(title: str, fig, attempts: int = 3, wait: float = 0.25):
    """Retry chart export multiple times (for slow file writes)."""
    last_err = None
    for i in range(attempts):
        path = save_chart_image(title, fig)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        last_err = path
        time.sleep(wait * (i + 1))
    return None