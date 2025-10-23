# ============================================
# utils/chart_saver.py — v4.3 | Kaleido Bypass (Guaranteed Color Export)
# ============================================
import os, io, time, base64
import streamlit as st
import plotly.express as px
from PIL import Image
from plotly.io import to_html
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

TMP_DIR = os.path.join("/tmp", "temp_charts")
os.makedirs(TMP_DIR, exist_ok=True)
PALETTE = px.colors.qualitative.Vivid

def _html_to_png(fig, out_path):
    """Render Plotly fig to PNG via headless Chrome (bypasses Kaleido)."""
    html_str = to_html(fig, include_plotlyjs="cdn", full_html=True)
    html_path = os.path.join(TMP_DIR, "temp_chart.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1600,900")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--hide-scrollbars")
    driver = webdriver.Chrome(options=opts)
    driver.get(f"file://{os.path.abspath(html_path)}")
    time.sleep(1.2)  # allow chart render
    png_data = driver.get_screenshot_as_png()
    driver.quit()

    im = Image.open(io.BytesIO(png_data))
    im.save(out_path, "PNG")
    return out_path


def save_chart_image(title, fig, filename_safe=None):
    """Universal Plotly → PNG export with color guarantee."""
    try:
        safe_name = (filename_safe or title).replace(" ", "_")
        out_path = os.path.join(TMP_DIR, f"{safe_name}.png")

        # Apply consistent bright layout
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
        )

        for i, trace in enumerate(fig.data):
            if trace.type == "pie":
                trace.marker.colors = PALETTE[: len(trace.labels)]
            elif hasattr(trace, "marker"):
                trace.marker.color = PALETTE[i % len(PALETTE)]

        return _html_to_png(fig, out_path)
    except Exception as e:
        st.error(f"⚠️ Chart render failed for '{title}': {e}")
        return None


# --------------------------------------------
# 🔁 Retry Wrapper
# --------------------------------------------
def ensure_chart_saved(title: str, fig, attempts: int = 3, wait: float = 0.25):
    """Retry chart export multiple times if Selenium render misbehaves."""
    last_err = None
    for i in range(attempts):
        path = save_chart_image(title, fig)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        last_err = path
        time.sleep(wait * (i + 1))
    return None