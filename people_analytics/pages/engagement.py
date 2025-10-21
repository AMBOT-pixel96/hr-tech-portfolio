# ============================================
# pages/engagement.py — v2.0
# ============================================

import streamlit as st
st.set_page_config(page_title="Engagement Analytics", layout="wide")

from utils.ui_styling import apply_sidebar_theme
from modules.engagement_module import run_engagement_module

# 🎨 Apply theme and launch module
apply_sidebar_theme()
run_engagement_module()