# ============================================
# utils/persistence_chatbot.py — v2.5 | Fusion Ready Edition
# ============================================
"""
Handles:
- Session persistence (save, load, auto-sync)
- Job-level sequencing (dropdown-based)
- Smart chatbot with cross-module awareness
- Supports correlations across modules (Insight Fusion Arc)
"""

import os, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime

# ==========================
# 🔹 Global Constants
# ==========================
SESSION_DIR = os.path.join(os.getcwd(), "session_data")
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE = os.path.join(SESSION_DIR, "people_analytics_state.json")

# ==========================
# 🧠 Persistence Bootstrap
# ==========================
def bootstrap_persistence():
    """Loads session data (job order + chatbot memory) on startup."""
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if k not in st.session_state:
                    st.session_state[k] = v
            st.caption(f"🧠 Memory auto-restored (last saved: {data.get('last_saved', 'unknown')})")
        else:
            st.caption("🚀 Fresh session started.")
    except Exception as e:
        st.warning(f"⚠️ Could not restore session: {e}")

# ==========================
# 💾 Save + Load Helpers
# ==========================
def save_session_state(filename=SESSION_FILE):
    """Manually save key session variables to disk."""
    try:
        data = {
            "job_order": st.session_state.get("job_order", []),
            "messages": st.session_state.get("messages", []),
            "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        st.success("💾 Session saved successfully.")
    except Exception as e:
        st.error(f"⚠️ Error saving session: {e}")

def load_session_state(filename=SESSION_FILE):
    """Manual restore if user clicks restore."""
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                st.session_state[k] = v
            st.success("🔁 Session restored successfully.")
        else:
            st.warning("⚠️ No session file found.")
    except Exception as e:
        st.error(f"⚠️ Error loading session: {e}")

# ==========================
# ⚙️ Job Level Sequencer
# ==========================
def job_level_sequencer_ui(emp_df: pd.DataFrame):
    """Interactive dropdown-based job hierarchy ranking."""
    if emp_df is None or "JobLevel" not in emp_df.columns:
        st.warning("⚠️ No JobLevel column found in dataset.")
        return

    st.subheader("⚙️ Step — Define Job Level Hierarchy")
    job_levels = sorted(emp_df["JobLevel"].dropna().unique().tolist())

    if "job_order" not in st.session_state:
        st.session_state.job_order = job_levels

    ranked_levels = {}
    for level in job_levels:
        rank = st.selectbox(
            f"Select hierarchy rank for: {level}",
            options=list(range(1, len(job_levels) + 1)),
            index=job_levels.index(level),
            key=f"rank_{level}"
        )
        ranked_levels[level] = rank

    if st.button("✅ Apply Order", use_container_width=True):
        ordered = [lvl for lvl, r in sorted(ranked_levels.items(), key=lambda x: x[1])]
        st.session_state.job_order = ordered
        st.success(f"Hierarchy updated: {', '.join(ordered)}")

    default_order = [
        "Analyst", "Assistant Manager", "Manager", "Senior Manager",
        "Associate Partner", "Director", "Executive", "Senior Executive"
    ]
    if st.button("↩️ Restore Default Order", use_container_width=True):
        st.session_state.job_order = default_order
        st.success("Default order restored.")

    st.info(f"Current hierarchy:\n{', '.join(st.session_state.job_order)}")

# ==========================
# 🤖 Smart HR Chatbot (Fusion Mode)
# ==========================
def run_chatbot_ui(modules_data: dict, primary_table_key="compensation"):
    """Runs a global chatbot with cross-module correlation intelligence."""
    st.subheader("💬 Unified HR Chatbot — Insight Fusion Mode")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # --- Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Process new input
    if user_input := st.chat_input("Ask me anything (e.g., 'Compare attrition vs engagement')"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        response = generate_smart_response(user_input.lower(), modules_data, primary_table_key)
        st.session_state["messages"].append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

    # --- Auto-save chat memory
    try:
        data = {
            "messages": st.session_state.get("messages", []),
            "job_order": st.session_state.get("job_order", []),
            "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(SESSION_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# ==========================
# 🧠 Insight Fusion Logic
# ==========================
def generate_smart_response(prompt: str, modules_data: dict, primary_key: str):
    """Handles natural queries and performs cross-module correlations."""
    icon = "🤖"
    try:
        # === Cross-Module Fusion Examples ===
        if "attrition" in prompt and "engagement" in prompt:
            if "attrition" in modules_data and "engagement" in modules_data:
                attr_df = modules_data["attrition"].copy()
                eng_df = modules_data["engagement"].copy()

                # Standardize department key
                if "Department" in attr_df.columns and "Department" in eng_df.columns:
                    merged = pd.merge(
                        attr_df, eng_df, on="Department", suffixes=("_attr", "_eng")
                    )
                    if "AttritionFlag" in merged.columns and "EngagementIndex" in merged.columns:
                        merged["AttritionBinary"] = merged["AttritionFlag"].map({"Yes": 1, "No": 0})
                        corr = merged["AttritionBinary"].corr(merged["EngagementIndex"])
                        return f"{icon} Correlation between engagement and attrition: **{corr:.2f}** (negative → higher engagement reduces attrition)"
            return f"{icon} I couldn’t find both engagement and attrition data with comparable fields."

        # === Compensation vs Performance
        if "compensation" in prompt and "performance" in prompt:
            if "compensation" in modules_data and "performance" in modules_data:
                comp = modules_data["compensation"].copy()
                perf = modules_data["performance"].copy()

                merged = pd.merge(comp, perf, on="EmployeeID", how="inner")
                if "CTC" in merged.columns and "PerformanceRating" in merged.columns:
                    corr = merged["CTC"].corr(merged["PerformanceRating"])
                    return f"{icon} Compensation vs Performance correlation: **{corr:.2f}** — positive indicates higher pay for higher performance."

        # === Gender Pay Gap
        if "gender" in prompt and "pay" in prompt:
            if "compensation" in modules_data:
                df = modules_data["compensation"]
                if {"Gender", "CTC"}.issubset(df.columns):
                    pivot = df.groupby("Gender")["CTC"].mean().round(2)
                    gap = (pivot.max() - pivot.min()) / pivot.max() * 100
                    return f"{icon} Gender pay gap: {gap:.1f}% — {pivot.idxmax()}s earn more on average."

        # === Engagement Summary
        if "engagement" in prompt:
            if "engagement" in modules_data:
                df = modules_data["engagement"]
                if "EngagementIndex" in df.columns:
                    avg = df["EngagementIndex"].mean().round(2)
                    return f"{icon} Average engagement index across organization: **{avg} / 5**"

        # === Attrition Summary
        if "attrition" in prompt:
            if "attrition" in modules_data:
                df = modules_data["attrition"]
                if "AttritionFlag" in df.columns:
                    attr_rate = (df["AttritionFlag"].eq("Yes").mean() * 100).round(2)
                    return f"{icon} Overall attrition rate: **{attr_rate}%**"

        # === Compensation Overview
        if "ctc" in prompt or "salary" in prompt or "compensation" in prompt:
            if "compensation" in modules_data:
                df = modules_data["compensation"]
                if "CTC" in df.columns:
                    avg_ctc = df["CTC"].mean() / 1e5
                    return f"{icon} Average CTC across organization: ₹{avg_ctc:.2f} Lakhs"

        # === Default
        return f"{icon} I couldn’t find an exact match. Try queries like:\n• 'Compare attrition vs engagement'\n• 'Gender pay gap'\n• 'Compensation vs performance'"

    except Exception as e:
        return f"{icon} Error while analyzing: {e}"