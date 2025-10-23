# ============================================
# utils/chart_saver.py — v4.0 | InMemory Plotly -> PNG exporter
# ============================================
import os
import time
import io
import traceback
import streamlit as st

try:
    import plotly.io as pio
except Exception:
    pio = None

try:
    from PIL import Image
except Exception:
    Image = None

# directory where PNGs are written for embedding
TMP_DIR = "temp_charts"
os.makedirs(TMP_DIR, exist_ok=True)

def _write_bytes_to_file(b: bytes, path: str):
    """Write bytes to path atomically."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # tiny delay to ensure FS visibility
    time.sleep(0.12)
    return path

def _ensure_png_has_white_bg_from_bytes(b: bytes, out_path: str):
    """
    Use Pillow to composite PNG bytes over white background and save to out_path.
    Returns out_path on success, None on failure.
    """
    if Image is None:
        # fallback: write bytes directly
        try:
            _write_bytes_to_file(b, out_path)
            return out_path
        except Exception as e:
            st.warning(f"⚠️ Pillow not available and direct write failed: {e}")
            return None

    try:
        with Image.open(io.BytesIO(b)) as im:
            # convert RGBA => RGB over white bg
            if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                bg.paste(im, mask=im.split()[-1])
                bg.save(out_path, "PNG")
            else:
                # ensure RGB
                im.convert("RGB").save(out_path, "PNG")
        # small pause
        time.sleep(0.08)
        return out_path
    except Exception as e:
        st.warning(f"⚠️ PNG post-process failed: {e}")
        # fallback to raw bytes
        try:
            _write_bytes_to_file(b, out_path)
            return out_path
        except Exception as ex:
            st.error(f"⚠️ Fallback write failed: {ex}")
            return None

def save_chart_image(title: str, fig, filename_safe: str = None, width: int = 1200, height: int = 700, scale: int = 2):
    """
    Export a Plotly figure to a PNG on disk and return the path.
    Strategy:
      1) Try fig.to_image(format='png') (kaleido) in-memory
      2) Post-process with Pillow to enforce RGB white background (removes transparency)
      3) Write final PNG to TMP_DIR with atomic replace
      4) Return path or None
    Notes:
      - Uses in-memory bytes to avoid partial file writes visible to ReportLab
      - Waits a tiny amount to ensure file system propagation
    """
    try:
        safe_name = (filename_safe or title).replace(" ", "_").replace("/", "_")
        out_path = os.path.join(TMP_DIR, f"{safe_name}.png")
        # ---- 1) in-memory image export ----
        image_bytes = None
        # Use Plotly's to_image (kaleido) if available
        try:
            if pio is None:
                raise RuntimeError("plotly.io not available")
            # prefer fig.to_image (most consistent)
            image_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
            if not image_bytes or len(image_bytes) < 10:
                raise RuntimeError("Empty image bytes returned")
        except Exception as e:
            # Log and bubble up to attempt other strategies
            st.warning(f"⚠️ plotly in-memory export failed for '{title}': {e}")
            # Try pio.to_image fallback
            try:
                image_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
            except Exception as e2:
                st.warning(f"⚠️ plotly.io.to_image fallback failed: {e2}")
                image_bytes = None

        if image_bytes is None:
            st.warning(f"⚠️ Could not generate PNG bytes for '{title}' (kaleido likely failing).")
            return None

        # ---- 2) Post-process & write atomically ----
        final_path = _ensure_png_has_white_bg_from_bytes(image_bytes, out_path)
        if final_path is None or not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            st.warning(f"⚠️ Saving PNG for '{title}' failed or file empty.")
            return None

        # Double-check size to avoid 0-byte
        if os.path.getsize(final_path) < 200:
            st.warning(f"⚠️ PNG filesize suspiciously small for '{title}'.")
            return final_path

        return final_path

    except Exception as e:
        st.error(f"⚠️ save_chart_image exceptional error for '{title}': {e}")
        st.debug(traceback.format_exc())
        return None

# Small helper used by PDF builder to ensure chart path is ready
def ensure_chart_saved(title: str, fig, attempts: int = 3, wait: float = 0.25):
    """
    Try to save chart multiple times (sometimes kaleido/IO transient failures occur).
    Returns path or None.
    """
    last_err = None
    for i in range(attempts):
        path = save_chart_image(title, fig)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        last_err = path
        time.sleep(wait * (i + 1))
    return None