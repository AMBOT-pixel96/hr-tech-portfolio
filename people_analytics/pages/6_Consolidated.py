# ============================================
# pages/6_Consolidated.py — Launcher Wrapper
# ============================================

import streamlit as st
from modules import consolidated_module

# Simply rerun consolidated module from the main page context
consolidated_module  # just ensure import happens

st.session_state["page_origin"] = "sidebar"
st.experimental_rerun()