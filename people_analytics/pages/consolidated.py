# ============================================
# pages/consolidated.py — v1.0 | HR Leadership Report Hub
# ============================================

import streamlit as st
from modules.consolidated_module import run_consolidated_module

st.set_page_config(
    page_title="Consolidated HR Executive Report",
    page_icon="📚",
    layout="wide"
)

# ---- HEADER ----
st.markdown("""
<div style="padding:16px;border-radius:10px;background:linear-gradient(90deg,#0F172A,#1E293B);color:white;">
  <h1 style="margin:0;font-size:26px;">📚 Consolidated HR Executive Reporting Engine</h1>
  <p style="margin-top:6px;">Generate a unified, leadership-ready report from all HR datasets (Performance, Attrition, Compensation, Workforce & Engagement).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---- MAIN APP ----
run_consolidated_module()

# ---- FOOTER ----
st.markdown("---")
st.caption("""
<small>
💡 <b>Tip:</b> Upload all 5 datasets and click <b>Generate Consolidated HR Executive PDF</b>  
to create a single, presentation-ready HR Insights deck for leadership.
</small>
""", unsafe_allow_html=True)