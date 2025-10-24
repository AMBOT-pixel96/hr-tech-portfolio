# ============================================
# pages/consolidated.py — v3.0 | Executive Stable Entry
# ============================================
"""
📘 Consolidated HR Leadership Deck Entry Point
------------------------------------------------
This lightweight entry script simply loads the fully-featured
module located at `modules/consolidated_module.py`.

✅ Keeps sidebar styling and global config consistent
✅ Avoids duplicate set_page_config() calls
✅ Prevents Streamlit reload loops
✅ Ensures lightning-fast load time
"""

import streamlit as st

# -------------------------------------------------------
# 🧭 Page Identity (only meta info, not UI config)
# -------------------------------------------------------
st.set_page_config(
    page_title="Consolidated HR Leadership Deck",
    page_icon="📘",
    layout="wide"
)

# -------------------------------------------------------
# 🧠 Safety Wrapper — prevents multiple reload loops
# -------------------------------------------------------
try:
    # import modules.consolidated_module
  import modules.consolidated_module_diagnostic
except Exception as e:
    st.error("⚠️ Failed to load Consolidated HR Deck module.")
    st.exception(e)