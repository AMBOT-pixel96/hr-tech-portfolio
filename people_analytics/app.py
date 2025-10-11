# ============================================
# app.py — People Analytics Dashboard (v1.0)
# ============================================

import streamlit as st
import os
import json
from datetime import datetime

# ---------------------------
# Global Config
# ---------------------------
st.set_page_config(
    page_title="People Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

# Ensure persistent directory exists
SESSION_DIR = os.path.join(os.getcwd(), "session_data")
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE = os.path.join(SESSION_DIR, "people_analytics_state.json")

# ---------------------------
# Persistence Utilities
# ---------------------------
def preload_session_state(filename=SESSION_FILE):
    """Restore previously saved session variables."""
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if k not in st.session_state:
                    st.session_state[k] = v
            st.caption("🧠 Memory restored from previous session.")
        else:
            st.caption("🚀 Fresh session started.")
    except Exception as e:
        st.warning(f"⚠️ Could not restore session: {e}")

def auto_save_session_state(filename=SESSION_FILE):
    """Auto-save session variables."""
    try:
        data = {k: v for k, v in st.session_state.items() if not k.startswith("_")}
        data["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Auto-save skipped: {e}")

# Load existing session state early
preload_session_state()

# ---------------------------
# Custom Styling
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
h1, h2, h3, h4 {
    color: #F9FAFB;
}
.tile {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    transition: transform 0.2s ease-in-out;
    border: 1px solid #1F2937;
    background: linear-gradient(180deg,#1E293B 0%,#0F172A 100%);
}
.tile:hover {
    transform: scale(1.03);
    border-color: #3B82F6;
}
.tile h3 {
    color: #FACC15;
}
.metric-box {
    font-size: 22px;
    font-weight: bold;
    color: #93C5FD;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div style='text-align:center; margin-top:20px;'>
    <h1>📊 People Analytics Dashboard</h1>
    <p style='color:#9CA3AF;'>A unified suite for HR insights across performance, engagement, pay, and workforce strategy.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Scorecard Section (Mock placeholders for now)
# ---------------------------
st.markdown("### 🔎 Executive Summary — Key Metrics Overview")

cols = st.columns(5)
scorecards = {
    "Performance Index": "78%",
    "Engagement Index": "4.2 / 5",
    "Compensation Fairness": "+3.4% gender gap",
    "Attrition Rate": "12.7%",
    "Workforce Balance": "1:6 span ratio"
}

for idx, (metric, value) in enumerate(scorecards.items()):
    with cols[idx]:
        st.markdown(f"""
        <div class='tile'>
            <h3>{metric}</h3>
            <div class='metric-box'>{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Tiles Navigation Section
# ---------------------------
st.markdown("---")
st.markdown("### 🧭 Explore Analytics Modules")

tile_cols = st.columns(5)
tiles = [
    ("📈 Performance", "Analyze rating distribution, pay vs performance, skill correlation.", "modules/performance.py"),
    ("💬 Engagement", "Upload survey data, measure engagement, identify hot-zones.", "modules/engagement.py"),
    ("💰 Compensation", "Analyze pay fairness, bonus distribution, and market benchmarking.", "modules/compensation.py"),
    ("📉 Attrition", "Explore exit trends, tenure analysis, and attrition hotspots.", "modules/attrition.py"),
    ("🏢 Workforce & Talent", "Assess structure, spans, and skill inventory analytics.", "modules/workforce.py")
]

for idx, (title, desc, path) in enumerate(tiles):
    with tile_cols[idx]:
        if st.button(title, use_container_width=True):
            st.session_state["active_module"] = path
            st.session_state["last_clicked"] = title
            auto_save_session_state()
            st.switch_page(path)
        st.caption(desc)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:13px; color:#9CA3AF;'>
Prepared with ❤️ by <a href='https://www.linkedin.com/in/amlan-mishra-7aa70894' target='_blank' style='color:#60A5FA;'>Amlan Mishra</a> |
<a href='https://github.com/AMBOT-pixel96/hr-tech-portfolio' target='_blank' style='color:#60A5FA;'>GitHub Portfolio</a>
</div>
""", unsafe_allow_html=True)