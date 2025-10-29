# ============================================
# utils/persistence_chatbot.py — v2.6 | Resilient Fusion + Safe Chatbot
# ============================================
import os, json
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px

SESSION_DIR = os.path.join(os.getcwd(), "session_data")
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE = os.path.join(SESSION_DIR, "people_analytics_state.json")

# ------------------------------
# Persistence Bootstrap
# ------------------------------
def bootstrap_persistence():
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

# ------------------------------
# Job Level Sequencer
# ------------------------------
def job_level_sequencer_ui(emp_df: pd.DataFrame):
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

    if st.button("↩️ Restore Default Order", use_container_width=True):
        default_order = ["Analyst", "Assistant Manager", "Manager", "Senior Manager", "Director"]
        st.session_state.job_order = default_order
        st.success("Default order restored.")

    st.info(f"Current hierarchy: {', '.join(st.session_state.job_order)}")

# ------------------------------
# Smart Chatbot
# ------------------------------
def run_chatbot_ui(modules_data: dict, primary_table_key="compensation"):
    st.subheader("💬 Unified HR Chatbot — Insight Fusion Mode")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask me anything (e.g., 'Compare attrition vs engagement')"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        response = generate_smart_response(user_input.lower(), modules_data, primary_table_key)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# ------------------------------
# Smart Response Generator
# ------------------------------
def generate_smart_response(prompt: str, modules_data: dict, primary_key: str):
    icon = "🤖"
    modules_data = {k: v for k, v in modules_data.items() if isinstance(v, pd.DataFrame) and not v.empty}
    if not modules_data:
        return f"{icon} No active module data detected. Please ensure datasets are loaded."

    try:
        import re
        p = prompt.lower()
        def has(*words): return any(re.search(rf"\\b{w}\\b", p) for w in words)

        # Compensation ↔ Performance
        if has("performance", "rating") and has("compensation", "ctc"):
            if "compensation" in modules_data and "performance" in modules_data:
                c, p_df = modules_data["compensation"], modules_data["performance"]
                if {"EmployeeID", "CTC"}.issubset(c.columns) and {"EmployeeID", "PerformanceRating"}.issubset(p_df.columns):
                    m = c.merge(p_df, on="EmployeeID", how="inner")
                    corr = m["CTC"].corr(m["PerformanceRating"])
                    fig = px.scatter(m, x="PerformanceRating", y="CTC", trendline="ols", title="Compensation vs Performance")
                    st.plotly_chart(fig, use_container_width=True)
                    return f"{icon} Compensation ↔ Performance correlation: **{corr:.2f}**."

        # Gender pay gap
        if has("gender") and has("ctc", "pay", "compensation"):
            if "compensation" in modules_data:
                df = modules_data["compensation"]
                if {"Gender", "CTC"}.issubset(df.columns):
                    g = df.groupby("Gender")["CTC"].mean().round(2)
                    gap = (g.max() - g.min()) / g.max() * 100
                    fig = px.bar(g, x=g.index, y=g.values, title="Average CTC by Gender")
                    st.plotly_chart(fig, use_container_width=True)
                    return f"{icon} Gender pay gap: **{gap:.1f}%** — {g.idxmax()}s earn more."

        return f"{icon} I couldn’t interpret that yet — try: Compare attrition vs engagement · Compensation vs performance · Gender pay gap"
    except Exception as e:
        return f"{icon} Error: {e}"