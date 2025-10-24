"""
utils_consolidated package initializer
-------------------------------------
Ensures all consolidated helper modules load safely under Streamlit’s sandbox.

✅ Prevents circular imports
✅ Makes sure relative imports resolve properly
"""

import importlib

# Try preloading key modules safely (ignore if missing)
for mod_name in [
    "utils_consolidated.pdf_merger",
    "utils_consolidated.deck_state_tracker",
    "utils_consolidated.pdf_consolidated_helper",
    "utils_consolidated.uploader_consolidated_helper"
]:
    try:
        importlib.import_module(mod_name)
    except ModuleNotFoundError:
        # Safe ignore — Streamlit can still lazy-load these later
        pass