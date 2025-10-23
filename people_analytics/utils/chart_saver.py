# ============================================
# utils/chart_saver.py — v4.1 | Full Fidelity Color + Atomic Export
# ============================================
import os
import time
import io
import traceback
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

# Directory for exported chart images
TMP_DIR = "temp_charts"
os.makedirs(TMP_DIR, exist_ok=True)

# Default bright color palette
PALETTE = px.colors.qualitative.Vivid

def _write_bytes_to_file(b: bytes, path: str):
    """Atomic write helper for bytes -> file."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    time.sleep(0.12)
    return path


def _ensure_png_has_white_bg_from_bytes(b: bytes, out_path: str):
    """Composites transparent PNGs over white background."""
    if Image is None:
        try:
            _write_bytes_to_file(b, out_path)
            return out_path
        except Exception as e:
            st.warning(f"⚠️ Pillow not available and direct write failed: {e}")
            return None

    try:
        with Image.open(io.BytesIO(b)) as im:
            if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                bg.paste(im, mask=im.split()[-1])
                bg.save(out_path, "PNG")
            else:
                im.convert("RGB").save(out_path, "PNG")
        time.sleep(0.08)
        return out_path
    except Exception as e:
        st.warning(f"⚠️ PNG post-process failed: {e}")
        try:
            _write_bytes_to_file(b, out_path)
            return out_path
        except Exception as ex:
            st.error(f"⚠️ Fallback write failed: {ex}")
            return None


def _apply_color_fidelity_fix(fig):
    """
    Ensures vivid categorical colors and strong contrast for Kaleido rendering.
    Fixes greyscale/black artifacts for bar, scatter, AND pie charts.
    """
    try:
        # Force bright layout
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
        )

        for i, trace in enumerate(fig.data):
            color_idx = i % len(PALETTE)

            # Bar / Scatter / Line traces
            if hasattr(trace, "marker") and hasattr(trace.marker, "color"):
                if getattr(trace.marker, "color", None) in [None, "#000", "black"]:
                    trace.marker.color = PALETTE[color_idx]
                trace.marker.line = dict(width=0.8, color="#333333")

            # Pie traces → use marker.colors (plural)
            if trace.type == "pie":
                trace.marker.colors = PALETTE[: len(trace.labels)]
                trace.marker.line = dict(width=1, color="#FFFFFF")

            # Line traces
            if hasattr(trace, "line") and hasattr(trace.line, "color"):
                if getattr(trace.line, "color", None) in [None, "#000", "black"]:
                    trace.line.color = PALETTE[color_idx]

    except Exception as e:
        st.warning(f"⚠️ Color fidelity patch failed: {e}")


def save_chart_image(title: str, fig, filename_safe: str = None, width: int = 1200, height: int = 700, scale: int = 2):
    """
    Export a Plotly figure to PNG with full color retention and atomic write.
    """
    try:
        safe_name = (filename_safe or title).replace(" ", "_").replace("/", "_")
        out_path = os.path.join(TMP_DIR, f"{safe_name}.png")

        # 🧠 Apply color correction
        _apply_color_fidelity_fix(fig)

        # ---- 1) In-memory image export (Kaleido) ----
        image_bytes = None
        try:
            if pio is None:
                raise RuntimeError("plotly.io not available")
            image_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
            if not image_bytes or len(image_bytes) < 10:
                raise RuntimeError("Empty image bytes returned")
        except Exception as e:
            st.warning(f"⚠️ plotly in-memory export failed for '{title}': {e}")
            try:
                image_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
            except Exception as e2:
                st.warning(f"⚠️ plotly.io.to_image fallback failed: {e2}")
                image_bytes = None

        if image_bytes is None:
            st.warning(f"⚠️ Could not generate PNG bytes for '{title}' (Kaleido likely failing).")
            return None

        # ---- 2) Post-process with Pillow for white background ----
        final_path = _ensure_png_has_white_bg_from_bytes(image_bytes, out_path)
        if final_path is None or not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            st.warning(f"⚠️ Saving PNG for '{title}' failed or file empty.")
            return None

        if os.path.getsize(final_path) < 200:
            st.warning(f"⚠️ PNG filesize suspiciously small for '{title}'.")
            return final_path

        return final_path

    except Exception as e:
        st.error(f"⚠️ save_chart_image exceptional error for '{title}': {e}")
        st.debug(traceback.format_exc())
        return None


def ensure_chart_saved(title: str, fig, attempts: int = 3, wait: float = 0.25):
    """Retry chart export multiple times if Kaleido misbehaves."""
    last_err = None
    for i in range(attempts):
        path = save_chart_image(title, fig)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        last_err = path
        time.sleep(wait * (i + 1))
    return None