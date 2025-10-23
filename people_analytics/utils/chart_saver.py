# ============================================
# utils/chart_saver.py — v3.5 | Color Safe + Unicode Ready
# ============================================
import os, time, plotly.io as pio, streamlit as st
from plotly.io._kaleido import scope

# 🧩 Ensure color-safe Kaleido export
scope.default_format = "png"
scope.default_width = 1200
scope.default_height = 700
scope.default_scale = 2
scope.mathjax = None
pio.kaleido.scope.chromium_args += ("--force-color-profile=srgb", "--disable-gpu")

TMP_DIR = "temp_charts"
os.makedirs(TMP_DIR, exist_ok=True)


def save_chart_image(title, fig):
    """Export Plotly fig as high-quality color PNG, fully flushed before use."""
    try:
        img_path = os.path.join(TMP_DIR, f"{title.replace(' ', '_')}.png")

        # Always force white background to retain color contrast
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
        )

        # ✅ Force Kaleido color rendering
        pio.write_image(fig, img_path, format="png", scale=2, engine="kaleido")

        # Wait until the file is fully flushed
        for _ in range(5):
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                break
            time.sleep(0.3)

        if not os.path.exists(img_path):
            raise IOError("Chart save failed — file not found after write.")

        return img_path

    except Exception as e:
        st.warning(f"⚠️ Chart save error for {title}: {e}")
        return None