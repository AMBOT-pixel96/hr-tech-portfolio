# ============================================
# pages/workforce.py — Workforce & Talent Analytics
# ============================================

import streamlit as st
from modules.workforce_module import run_workforce_module
from utils.ui_styling import apply_sidebar_theme

# ⚙️ Page config MUST come before any Streamlit output
st.set_page_config(page_title="Workforce & Talent Analytics", layout="wide")

# 🎨 Apply Executive Sidebar Theme (no need to duplicate CSS below)
apply_sidebar_theme()

# 🧩 Run the module
run_workforce_module()