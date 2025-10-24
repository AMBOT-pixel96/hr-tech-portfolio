"""
utils package initializer
-------------------------
Ensures core utility modules load cleanly inside Streamlit
and prevents circular import crashes during app startup.
"""

import importlib

# Preload core helpers that are used across modules
for mod_name in [
    "utils.pdf_helper",
    "utils.uploader_helper",
    "utils.chart_saver",
    "utils.template_helper",
]:
    try:
        importlib.import_module(mod_name)
    except ModuleNotFoundError:
        # Optional: some utilities might not exist in all builds
        pass