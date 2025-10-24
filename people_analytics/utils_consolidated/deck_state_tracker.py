# ============================================
# utils_consolidated/deck_state_tracker.py — v2.0 | Executive Stable (Final)
# ============================================
"""
Handles persistence of module addition timestamps for the
Consolidated HR Leadership Deck (stored in /tmp/deck_state.json).

✅ Tracks which module PDFs are added
✅ Saves timestamp in ISO format
✅ Auto-loads state safely on reboot
✅ Used by consolidated.py to show ✅ / ❌ with last updated date
"""

import os
import json
from datetime import datetime

# ----------------------------------------------------------------
# 📂 Storage Location
# ----------------------------------------------------------------
STATE_FILE = os.path.join("/tmp", "deck_state.json")

# ----------------------------------------------------------------
# 🧩 Load deck state safely
# ----------------------------------------------------------------
def get_module_state() -> dict:
    """Returns a dict of module timestamps (e.g., {'Attrition': '2025-10-24T06:23:00'})."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            # Ensure valid dictionary
            if isinstance(data, dict):
                return data
        return {}
    except Exception:
        return {}

# ----------------------------------------------------------------
# 💾 Update / add module entry
# ----------------------------------------------------------------
def update_module_state(module_name: str):
    """Updates (or adds) a module's timestamp when it’s added to the consolidated deck."""
    try:
        data = get_module_state()
        data[module_name] = datetime.now().isoformat()
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to update deck state: {e}")

# ----------------------------------------------------------------
# 🧹 Optional helper to clear deck state
# ----------------------------------------------------------------
def clear_module_state():
    """Resets all stored timestamps."""
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    except Exception as e:
        print(f"⚠️ Failed to clear deck state: {e}")