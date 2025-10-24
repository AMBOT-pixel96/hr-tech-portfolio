# ============================================
# app.py — People Analytics Dashboard (v3.1 Executive Stable)
# ============================================
# ===============================
# 🧠 Stability Patch — Disable Watchdog
# ===============================
import os
os.environ["STREAMLIT_WATCHDOG"] = "false"
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
    content: "📊 PEOPLE ANALYTICS DASHBOARD";
    margin-left: 20px;
    font-weight: 800;
    font-size: 16px;
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
a[href*="consolidated"] span::before { content: "📘 "; }
a[href*="app"] span::before { content: "🏠 "; }

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
.tile {
    padding: 25px; border-radius: 15px; text-align: center;
    transition: transform 0.2s ease-in-out;
    border: 1px solid #1F2937;
    background: linear-gradient(180deg,#1E293B 0%,#0F172A 100%);
    box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
}
.tile:hover { transform: scale(1.03); border-color: #3B82F6;
    box-shadow: 0px 0px 15px rgba(59,130,246,0.3);
}
.tile h3 { color: #FACC15; margin-bottom: 10px; }
.tile p { color: #CBD5E1; font-size: 14px; }

/* Golden Tile (Consolidated) */
.tile-gold {
    border: 1px solid #FACC15;
    background: linear-gradient(180deg,#1F2937 0%,#111827 100%);
    box-shadow: 0px 0px 12px rgba(250,204,21,0.25);
}
.tile-gold:hover {
    border-color: #FDE047;
    box-shadow: 0px 0px 18px rgba(250,204,21,0.35);
    transform: scale(1.03);
}
.tile-gold h3 { color: #FDE047; }
.tile-gold p { color: #E5E7EB; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 🏠 Header
# ---------------------------
st.markdown("""
<div style='text-align:center; margin-top:20px;'>
    <h1>📊 People Analytics Dashboard</h1>
    <p style='color:#9CA3AF;'>Unified HR suite for analytics across Workforce, Performance, Engagement, Compensation, and Attrition.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# ⚡ Navigation Tiles
# ---------------------------
st.markdown("---")
st.markdown("### ⚡ Explore Analytics Modules")

tiles = [
    {"icon": "🏢", "title": "Workforce Analytics", "desc": "Headcount, span & structure", "path": "pages/workforce.py"},
    {"icon": "🏆", "title": "Performance Analytics", "desc": "Understand how performance scores drive pay and progression.", "path": "pages/performance.py"},
    {"icon": "💬", "title": "Engagement Analytics", "desc": "Decode engagement and sentiment trends.", "path": "pages/engagement.py"},
    {"icon": "💰", "title": "Compensation Analytics", "desc": "Compare pay vs market and identify gender gaps.", "path": "pages/compensation.py"},
    {"icon": "📉", "title": "Attrition Analytics", "desc": "Explore turnover rates and tenure patterns.", "path": "pages/attrition.py"}
]

cols = st.columns(5)
for i, t in enumerate(tiles):
    with cols[i]:
        st.markdown(f"""
        <div class="tile">
            <h3>{t['icon']} {t['title']}</h3>
            <p>{t['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"{t['icon']} Open", key=f"tile_{i}", use_container_width=True):
            # ✅ Fixed: correct relative path, not capitalized route
            st.switch_page(t["path"])

# ---------------------------
# 📘 Consolidated HR Deck
# ---------------------------
st.markdown("---")
st.markdown("### 🧩 Leadership Deck")

st.markdown(f"""
<div class="tile-gold" style="margin:auto; width:80%;">
    <h3>📘 Consolidated HR Leadership Deck</h3>
    <p>Generate a single, boardroom-ready executive report combining all analytics modules into one golden PDF.</p>
</div>
""", unsafe_allow_html=True)

if st.button("📘 Open Consolidated HR Deck", use_container_width=True, key="deck_btn"):
    st.switch_page("pages/consolidated.py")

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