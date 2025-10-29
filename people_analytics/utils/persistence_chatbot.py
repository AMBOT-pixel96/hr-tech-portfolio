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
    """Handles natural queries + fuzzy matching + visual summaries."""
    import re, plotly.express as px
    icon = "🤖"
    p = prompt.lower()

    try:
        # --- Normalize some synonyms ---
        p = p.replace("salary", "ctc").replace("wage", "ctc")
        p = p.replace("bonus", "bonus %")

        # --- Helper: detect keywords loosely ---
        def has(*words): return any(re.search(rf"\b{w}\b", p) for w in words)

        # === 1️⃣ Attrition × Engagement correlation ===
        if has("attrition", "churn") and has("engagement"):
            if "attrition" in modules_data and "engagement" in modules_data:
                a, e = modules_data["attrition"], modules_data["engagement"]
                if "Department" in a and "Department" in e:
                    m = a.merge(e, on="Department", suffixes=("_attr", "_eng"))
                    if {"AttritionFlag", "EngagementIndex"}.issubset(m.columns):
                        m["AttritionBinary"] = m["AttritionFlag"].map({"Yes": 1, "No": 0})
                        corr = m["AttritionBinary"].corr(m["EngagementIndex"])
                        fig = px.scatter(m, x="EngagementIndex", y="AttritionBinary",
                                         trendline="ols", title="Engagement vs Attrition")
                        st.plotly_chart(fig, use_container_width=True)
                        return f"{icon} Correlation between engagement & attrition: **{corr:.2f}** (negative → higher engagement = lower attrition)"

        # === 2️⃣ Compensation × Performance correlation ===
        if has("performance", "rating") and has("compensation", "ctc"):
            if "performance" in modules_data and "compensation" in modules_data:
                c, p_df = modules_data["compensation"], modules_data["performance"]
                if "EmployeeID" in c and "EmployeeID" in p_df:
                    m = c.merge(p_df, on="EmployeeID", how="inner")
                    if {"CTC", "PerformanceRating"}.issubset(m.columns):
                        corr = m["CTC"].corr(m["PerformanceRating"])
                        fig = px.scatter(m, x="PerformanceRating", y="CTC",
                                         trendline="ols", title="Compensation vs Performance")
                        st.plotly_chart(fig, use_container_width=True)
                        return f"{icon} Compensation ↔ Performance correlation: **{corr:.2f}**."

        # === 3️⃣ Gender pay gap ===
        if has("gender") and has("ctc", "pay", "compensation"):
            if "compensation" in modules_data:
                df = modules_data["compensation"]
                if {"Gender", "CTC"}.issubset(df.columns):
                    g = df.groupby("Gender")["CTC"].mean().round(2)
                    gap = (g.max() - g.min()) / g.max() * 100
                    fig = px.bar(g, x=g.index, y=g.values, title="Average CTC by Gender")
                    st.plotly_chart(fig, use_container_width=True)
                    return f"{icon} Gender pay gap: **{gap:.1f}%** — {g.idxmax()}s earn more."

        # === 4️⃣ Average CTC by Rating ===
        if has("ctc") and has("rating", "performance"):
            if "performance" in modules_data:
                df = modules_data["performance"]
                if {"PerformanceRating", "CTC"}.issubset(df.columns):
                    avg = df.groupby("PerformanceRating")["CTC"].mean().round(0)
                    fig = px.bar(avg, x=avg.index, y=avg.values, title="Average CTC by Rating")
                    st.plotly_chart(fig, use_container_width=True)
                    return f"{icon} Average CTC by rating:\n{avg.to_string()}"

        # === 5️⃣ Engagement summary ===
        if has("engagement"):
            if "engagement" in modules_data and "EngagementIndex" in modules_data["engagement"].columns:
                avg = modules_data["engagement"]["EngagementIndex"].mean().round(2)
                return f"{icon} Average engagement index: **{avg}/5**."

        # === 6️⃣ Attrition summary ===
        if has("attrition", "churn"):
            if "attrition" in modules_data and "AttritionFlag" in modules_data["attrition"].columns:
                rate = modules_data["attrition"]["AttritionFlag"].eq("Yes").mean() * 100
                return f"{icon} Overall attrition rate: **{rate:.2f}%**."

        # === 7️⃣ Compensation overview ===
        if has("ctc", "compensation", "salary"):
            if "compensation" in modules_data and "CTC" in modules_data["compensation"].columns:
                mean = modules_data["compensation"]["CTC"].mean() / 1e5
                return f"{icon} Average CTC: ₹{mean:.2f} Lakhs."

        # --- Default help ---
        return f"{icon} I couldn't interpret that yet — try things like:\n• Compare attrition vs engagement\n• Compensation vs performance\n• Gender pay gap"

    except Exception as e:
        return f"{icon} Error: {e}"