# ============================================
# app.py — People Analytics Dashboard (v3.0 Command Center)
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
[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLinkActive"] {
    background: #1D4ED8;
    color: white !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 🧭 Session persistence
# ---------------------------
SESSION_DIR = os.path.join(os.getcwd(), "session_data")
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE = os.path.join(SESSION_DIR, "people_analytics_state.json")

def preload_session_state():
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if k not in st.session_state:
                    st.session_state[k] = v
            st.caption("🧠 Memory restored from previous session.")
        else:
            st.caption("🚀 Fresh session started.")
    except Exception as e:
        st.warning(f"⚠️ Session restore skipped: {e}")

def auto_save_session_state():
    try:
        data = {k: v for k, v in st.session_state.items() if not k.startswith("_")}
        data["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(SESSION_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Auto-save skipped: {e}")

preload_session_state()

# ---------------------------
# 🌑 Global Styling
# ---------------------------
st.markdown("""
<style>
body { background-color: #0E1117; color: white; }
h1, h2, h3, h4 { color: #F9FAFB; }

/* === Tile Cards === */
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
.tile h3 { color: #FACC15; margin-bottom: 10px; }
.tile p { color: #CBD5E1; font-size: 14px; }

/* === Consolidated Card === */
.consolidated-card {
    background: linear-gradient(135deg, #FACC15, #FBBF24);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    color: #1E1E1E;
    font-weight: 600;
    box-shadow: 0px 4px 12px rgba(255,215,0,0.4);
    transition: all 0.3s ease;
}
.consolidated-card:hover {
    transform: scale(1.03);
    box-shadow: 0px 6px 20px rgba(255,215,0,0.5);
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 🏠 Header
# ---------------------------
st.markdown("""
<div style='text-align:center; margin-top:20px;'>
    <h1>📊 People Analytics Command Center</h1>
    <p style='color:#9CA3AF;'>Unified HR suite for analytics across Workforce, Performance, Engagement, Compensation, and Attrition.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# ⚡ Navigation Tiles (Grid Layout)
# ---------------------------
st.markdown("---")
st.markdown("### ⚡ Explore Analytics Modules")

# 3 on Row 1
row1 = st.columns(3)
modules = [
    {"icon": "🏢", "title": "Workforce Analytics", "desc": "Headcount, span & structure", "path": "/1_Workforce"},
    {"icon": "🏆", "title": "Performance Analytics", "desc": "Distribution, KPIs & trends", "path": "/2_Performance"},
    {"icon": "💬", "title": "Engagement Analytics", "desc": "Sentiment & participation", "path": "/3_Engagement"},
]
for i, mod in enumerate(modules):
    with row1[i]:
        st.markdown(f"""
        <a href="{mod['path']}" target="_self" style="text-decoration:none;">
            <div class="tile">
                <h3>{mod['icon']} {mod['title']}</h3>
                <p>{mod['desc']}</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

# 2 on Row 2
row2 = st.columns(2)
modules2 = [
    {"icon": "💰", "title": "Compensation Analytics", "desc": "Pay, bonus & equity insights", "path": "/4_Compensation"},
    {"icon": "📉", "title": "Attrition Analytics", "desc": "Turnover & tenure trends", "path": "/5_Attrition"},
]
for i, mod in enumerate(modules2):
    with row2[i]:
        st.markdown(f"""
        <a href="{mod['path']}" target="_self" style="text-decoration:none;">
            <div class="tile">
                <h3>{mod['icon']} {mod['title']}</h3>
                <p>{mod['desc']}</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

# Consolidated Deck (Rectangular Golden)
st.markdown("---")
st.markdown(f"""
<a href="/_6_Consolidated" target="_self" style="text-decoration:none;">
    <div class="consolidated-card">
        <h2>📘 Generate HR Leadership Deck</h2>
        <p>Unified boardroom-ready PDF across all analytics modules</p>
    </div>
</a>
""", unsafe_allow_html=True)

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