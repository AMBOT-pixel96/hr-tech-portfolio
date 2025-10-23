# ============================================
# pages/consolidated.py — Main Consolidated HR Deck Page (Fixed)
# ============================================

import streamlit as st
import modules.consolidated_module  # Import auto-runs the module’s Streamlit logic

# -------------------------------------------------------
# 🧭 Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Consolidated HR Leadership Deck",
    page_icon="📘",
    layout="wide"
)

# -------------------------------------------------------
# 🎨 Unified Styling (for consistency only)
# -------------------------------------------------------
st.markdown("""
<style>
/* Sidebar theme consistency */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E3A8A 100%);
    color: white;
}

/* File upload card styling */
div[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, #1E293B, #0F172A) !important;
    border: 1px solid #1E3A8A !important;
    border-radius: 14px !important;
    padding: 18px !important;
    color: #E5E7EB !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #3B82F6 !important;
}

/* Buttons */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #1E3A8A, #2563EB);
    color: white !important;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(90deg, #2563EB, #1E40AF);
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)