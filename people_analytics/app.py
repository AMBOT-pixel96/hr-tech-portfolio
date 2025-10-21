# ============================================
# app.py — People Analytics Dashboard (v1.2 Executive Redesign)
# ============================================

import streamlit as st
import os
import json
import datetime

# ---------------------------
# Global Config
# ---------------------------
st.set_page_config(
    page_title="People Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------------------
# Sidebar Styling (Executive Theme + Glow Band)
# ---------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E3A8A 100%);
    color: white;
    padding-top: 1rem;
    border-right: 1px solid #1E293B;
}

/* --- Sidebar Title Band --- */
[data-testid="stSidebarNav"]::before {
    content: "📊 People Analytics Dashboard";
    margin-left: 20px;
    font-weight: 700;
    font-size: 18px;
    color: #FACC15;
    border-bottom: 2px solid #FACC15;
    padding-bottom: 6px;
    display: block;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* --- Sidebar Links --- */
[data-testid="stSidebarNav"] a {
    color: #E2E8F0 !important;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 15px;
    transition: all 0.25s ease-in-out;
    text-transform: capitalize;
    font-size: 15px;
    letter-spacing: 0.3px;
    text-decoration: none !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(255,255,255,0.1);
    transform: scale(1.03);
    text-shadow: 0 0 6px rgba(96,165,250,0.5);
}
[data-testid="stSidebarNav"] a[data-testid="stSidebarNavLinkActive"] {
    background: linear-gradient(90deg,#1D4ED8,#2563EB);
    color: white !important;
    font-weight: 700;
    box-shadow: 0 0 10px rgba(37,99,235,0.4);
}

/* --- Icons --- */
[data-testid="stSidebarNav"] a span::before { margin-right: 8px; }
a[href*="performance"] span::before { content: "🏆 "; }
a[href*="engagement"] span::before { content: "💬 "; }
a[href*="compensation"] span::before { content: "💰 "; }
a[href*="attrition"] span::before { content: "📉 "; }
a[href*="workforce"] span::before { content: "🏢 "; }
a[href*="app"] span::before { content: "🏠 "; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Session Persistence Setup
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
        data["last_saved"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Auto-save skipped: {e}")

# Load session early
preload_session_state()

# ---------------------------
# Global Styling
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
    animation: pulse 2.5s infinite;
}
@keyframes pulse {
  0% { color: #60A5FA; }
  50% { color: #93C5FD; }
  100% { color: #60A5FA; }
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
# Welcome Section
# ---------------------------
current_time = datetime.datetime.now().strftime("%A, %d %B %Y | %I:%M %p")

st.markdown(f"""
<div style="text-align:center; margin-top:20px; margin-bottom:20px;">
    <p style="font-size:16px; color:#A5B4FC;">🕒 {current_time}</p>
    <h3 style="color:#FACC15;">Welcome back, Amlan 👋</h3>
    <p style="color:#9CA3AF;">Choose a module below to begin analyzing your HR data.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 💡 Quick Start Tips
# ---------------------------
with st.expander("💡 Quick Start Guide — How to Use This App", expanded=False):
    st.markdown("""
    **Step 1:** Download the sample data template from any module.  
    **Step 2:** Upload your HR dataset (CSV or Excel).  
    **Step 3:** Explore insights, metrics, and visuals interactively.  
    **Step 4:** Export an **Executive PDF Report** for leadership review.  

    🧠 *Pro Tip:* Each module supports smart summaries — your data drives real-time insights.  
    """)

# ---------------------------
# Navigation Tiles
# ---------------------------
st.markdown("---")
st.markdown("### 🧭 Explore Analytics Modules")

tile_cols = st.columns(5)
tiles = [
    ("🏆 Performance", "Analyze rating distribution, pay vs performance, skill correlation.", "pages/performance.py"),
    ("💬 Engagement", "Upload survey data, measure engagement, identify hot-zones.", "pages/engagement.py"),
    ("💰 Compensation", "Analyze pay fairness, bonus distribution, and market benchmarking.", "pages/compensation.py"),
    ("📉 Attrition", "Explore exit trends, tenure analysis, and attrition hotspots.", "pages/attrition.py"),
    ("🏢 Workforce & Talent", "Assess structure, spans, and skill inventory analytics.", "pages/workforce.py")
]

for idx, (title, desc, path) in enumerate(tiles):
    with tile_cols[idx]:
        st.markdown(f"""
        <div class='tile'>
            <h3>{title}</h3>
            <p style='font-size:13px; color:#9CA3AF;'>{desc}</p>
            <form action='/{path}' target='_self'>
                <button style="
                    background: linear-gradient(90deg,#1E3A8A,#3B82F6);
                    border:none; border-radius:8px;
                    color:white; font-weight:600; padding:8px 16px;
                    cursor:pointer; transition:all 0.2s ease-in-out;
                " onmouseover="this.style.transform='scale(1.05)';"
                   onmouseout="this.style.transform='scale(1.00)';">
                    Launch
                </button>
            </form>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:13px; color:#9CA3AF;'>
Prepared with ❤️ by 
<a href='https://www.linkedin.com/in/amlan-mishra-7aa70894' target='_blank' style='color:#60A5FA;'>Amlan Mishra</a> |
<a href='https://github.com/AMBOT-pixel96/hr-tech-portfolio' target='_blank' style='color:#60A5FA;'>GitHub Portfolio</a>
</div>
""", unsafe_allow_html=True)