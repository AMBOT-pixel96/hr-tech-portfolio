# ============================================
# pages/performance.py — v2.0
# ============================================

import streamlit as st
st.set_page_config(page_title="Performance Analytics", layout="wide")

from utils.ui_styling import apply_sidebar_theme
from modules.performance_module import run_performance_module

# 🎨 Apply theme and launch module
apply_sidebar_theme()
run_performance_module()