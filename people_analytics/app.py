# ============================================
# app.py — People Analytics Dashboard (v1.4 Final Executive Edition)
# ============================================

import streamlit as st
import os
import json
from datetime import datetime

# ---------------------------
# 🧠 Global Config
# ---------------------------
st.set_page_config(
    page_title="People Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------------------
# 🎨 Sidebar Styling (Executive Theme)
# ---------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E3A8A 100%);
    color: white;
    padding-top: 1rem;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebarNav"]::before {
    content: "📊 People Analytics Dashboard";
    margin-left: 20px;
    font-weight: 700;
    font-size: 18px;
    color: #FACC15;
    text-transform: uppercase;
}
[data-testid="stSidebarNav"] a {
    color: #E2E8F0 !important;
    font-weight: 500;
    border-radius: 8px;
    padding: 10px 15px;
    transition: all 0.2s ease-in-out;
    text-transform: capitalize;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(255,255,255,0.1);
    transform: scale(1.03);
}
[data-testid="stSidebarNav"] a span::before { margin-right: 8px; }

a[href*="performance"] span::before { content: "🏆 "; }
a[href*="engagement"] span::before { content: "💬 "; }
a[href*="compensation"] span::before { content: "💰 "; }
a[href*="attrition"] span::before { content: "📉 "; }
a[href*="workforce"] span::before { content: "🏢 "; }
a[href*="app"] span::before { content: "🏠 "; }

[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLinkActive"] {
    background: #1D4ED8;
    color: white !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 🧩 Session Persistence Setup
# ---------------------------
SESSION_DIR = os.path.join(os.getcwd(), "session_data")
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE = os.path.join(SESSION_DIR, "people_analytics_state.json")

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

preload_session_state()

# ---------------------------
# 🌑 Global Styling (Dark Mode)
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
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    transition: transform 0.2s ease-in-out;
    border: 1px solid #1F2937;
    background: linear-gradient(180deg,#1E293B 0%,#0F172A 100%);
    box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
}
.tile:hover {
    transform: scale(1.03);
    border-color: #3B82F6;
    box-shadow: 0px 0px 15px rgba(59,130,246,0.3);
}
.tile h3 {
    color: #FACC15;
    margin-bottom: 10px;
}
.tile p {
    color: #CBD5E1;
    font-size: 14px;
}
.launch-btn {
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    color: white !important;
    padding: 8px 18px;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
}
.launch-btn:hover {
    background: linear-gradient(90deg, #3B82F6, #60A5FA);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 🏠 Header
# ---------------------------
st.markdown("""
<div style='text-align:center; margin-top:20px;'>
    <h1>📊 People Analytics Dashboard</h1>
    <p style='color:#9CA3AF;'>A unified suite for HR insights across Performance, Engagement, Compensation, Attrition, and Workforce Strategy.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 🧭 Analytics Modules Grid (Replaces Scorecards)
# ---------------------------
st.markdown("---")
st.markdown("### ⚡ Explore Analytics Modules")

tiles = [
    {"icon": "🏆", "title": "Performance Analytics", "desc": "Understand how performance scores drive pay and progression.", "path": "pages/performance.py"},
    {"icon": "💬", "title": "Engagement Analytics", "desc": "Decode employee sentiment and engagement trends.", "path": "pages/engagement.py"},
    {"icon": "💰", "title": "Compensation Analytics", "desc": "Compare internal pay vs market and identify fairness gaps.", "path": "pages/compensation.py"},
    {"icon": "📉", "title": "Attrition Analytics", "desc": "Explore turnover, tenure curves, and attrition hotspots.", "path": "pages/attrition.py"},
    {"icon": "🏢", "title": "Workforce Analytics", "desc": "Visualize spans, hierarchies, and skill distributions.", "path": "pages/workforce.py"}
]

cols = st.columns(5)
for i, t in enumerate(tiles):
    with cols[i]:
        if st.button(f"{t['icon']} {t['title']}", use_container_width=True, key=f"btn_{i}"):
            st.session_state["active_module"] = t["path"]
            auto_save_session_state()
            st.switch_page(t["path"])
        st.caption(t["desc"])

# ---------------------------
# ⚙️ Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:13px; color:#9CA3AF;'>
Prepared with ❤️ by 
<a href='https://www.linkedin.com/in/amlan-mishra-7aa70894' target='_blank' style='color:#60A5FA;'>Amlan Mishra</a> |
<a href='https://github.com/AMBOT-pixel96/hr-tech-portfolio' target='_blank' style='color:#60A5FA;'>GitHub Portfolio</a>
</div>
""", unsafe_allow_html=True)