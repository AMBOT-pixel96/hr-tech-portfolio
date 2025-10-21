# ============================================
# pages/engagement.py — Engagement Analytics
# ============================================

import streamlit as st
from modules.engagement_module import run_engagement_module
from utils.ui_styling import apply_sidebar_theme

# ⚙️ Page config MUST come before any Streamlit output
st.set_page_config(page_title="Engagement Analytics", layout="wide")

# 🎨 Apply Executive Sidebar Theme (no need to duplicate CSS below)
apply_sidebar_theme()

# 🧩 Run the module
run_engagement_module()