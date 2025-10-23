# ============================================
# utils/chart_saver.py — v3.0.1 | Dual-Theme + Sync Save Edition
# ============================================
import os
import time
import streamlit as st

def save_chart_image(title, fig):
    """
    Saves Plotly chart as high-quality PNG inside temp_charts directory.
    ✅ Keeps color fidelity in PDFs.
    ✅ Restores dark theme visuals inside the app.
    ✅ Waits for disk write completion to avoid broken PDFs.
    """
    try:
        TMP_DIR = "temp_charts"
        os.makedirs(TMP_DIR, exist_ok=True)
        img_path = os.path.join(TMP_DIR, f"{title.replace(' ', '_')}.png")

        # 🧠 Smart dual-theme handling:
        #  - White background & black text for PDF images
        #  - Retain original dark/light mode inside app
        orig_layout = fig.layout.to_plotly_json()  # backup layout before altering

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
        )

        # Save the figure (synchronously)
        fig.write_image(img_path, width=1200, height=700, scale=2)
        time.sleep(0.3)  # 🕒 small buffer to ensure file is ready

        # Restore original look for Streamlit display
        fig.update_layout(**orig_layout)

        # Double check file write
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            raise IOError("File save incomplete or empty.")

        return img_path

    except Exception as e:
        st.warning(f"⚠️ Could not save chart '{title}': {e}")
        return None