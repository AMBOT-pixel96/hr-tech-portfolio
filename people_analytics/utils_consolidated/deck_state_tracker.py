# ============================================
# utils_consolidated/deck_state_tracker.py — v1.0 | Module Update Tracker
# ============================================
import os
import json
from datetime import datetime

STATE_FILE = "/tmp/consolidated_pdfs/deck_state.json"
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

def _read_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def _write_state(data):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def update_module_state(module_name: str):
    """Updates timestamp when module was added."""
    state = _read_state()
    state[module_name] = datetime.now().isoformat()
    _write_state(state)

def get_module_state():
    """Returns module timestamps as dict."""
    return _read_state()