# ============================================
# pages/workforce.py — Workforce & Talent Analytics
# ============================================

import streamlit as st
from modules.workforce_module import run_workforce_module

# --- Unified Executive Sidebar Styling ---
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
}
[data-testid="stSidebarNav"] a {
    color: #E2E8F0 !important;
    font-weight: 500;
    border-radius: 8px;
    padding: 10px 15px;
    transition: all 0.2s ease-in-out;
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

st.set_page_config(page_title="Workforce & Talent Analytics", layout="wide")
run_workforce_module()