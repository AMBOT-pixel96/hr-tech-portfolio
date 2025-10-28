# utils/persistence_chatbot.py
# ============================================
# Persistence + JobLevel Sequencer + Chatbot
# (Drop into utils/ and import functions into consolidated page)
# ============================================

import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Config / Files
# ---------------------------
SESSION_DIR = os.path.join(os.getcwd(), "session_data")
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE = os.path.join(SESSION_DIR, "people_analytics_state.json")  # global session file

# ---------------------------
# Persistence: Load / Save
# ---------------------------
def preload_session_state(filename: str = SESSION_FILE):
    """Load session state if present, merging keys without overriding active ones."""
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if k not in st.session_state:
                    st.session_state[k] = v
            st.caption("🧠 Memory restored from previous session.")
        else:
            st.caption("🚀 Fresh session started.")
    except Exception as e:
        st.warning(f"⚠️ Could not restore session: {e}")


def save_session_state(filename: str = SESSION_FILE):
    """Save selective keys into the session file."""
    try:
        data = {
            "job_order": st.session_state.get("job_order", []),
            "messages": st.session_state.get("messages", []),
            "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        st.success(f"💾 Session saved to {filename}")
    except Exception as e:
        st.error(f"⚠️ Error saving session: {e}")


def load_session_state(filename: str = SESSION_FILE):
    """Explicit restore of persisted session state (write into st.session_state)."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            st.session_state[k] = v
        st.success(f"🔁 Session restored from {filename}")
    except FileNotFoundError:
        st.warning("No saved session found.")
    except Exception as e:
        st.error(f"⚠️ Error restoring session: {e}")


def auto_save_session_state(filename: str = SESSION_FILE, tracked_keys=None):
    """Auto-save when relevant session-state keys exist. Silent on failure."""
    if tracked_keys is None:
        tracked_keys = ["job_order", "messages"]
    try:
        if any(k in st.session_state for k in tracked_keys):
            data = {
                "job_order": st.session_state.get("job_order", []),
                "messages": st.session_state.get("messages", []),
                "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            # keep it subtle (no success pop for autosave)
            return True
    except Exception:
        pass
    return False


def auto_load_session_state(filename: str = SESSION_FILE):
    """Silently load session if present. Merge without overwriting runtime keys."""
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if k not in st.session_state:
                    st.session_state[k] = v
            st.info(f"🔁 Auto-restored session from {filename} (last saved: {data.get('last_saved','unknown')})", icon="🧠")
            return True
        else:
            st.caption("🧩 No previous session found — starting fresh.")
            return False
    except Exception as e:
        st.warning(f"⚠️ Auto-load skipped: {e}")
        return False

# ---------------------------
# Job Level Sequencer UI (call from consolidated module)
# ---------------------------
DEFAULT_JOB_ORDER = [
    "Analyst", "Assistant Manager", "Manager", "Senior Manager",
    "Associate Partner", "Director", "Executive", "Senior Executive"
]


def job_level_sequencer_ui(emp_df: pd.DataFrame | None = None, default_order=None):
    """
    Renders the Job Level Sequencer UI.
    - emp_df: dataframe with JobLevel column (optional, but recommended)
    - default_order: list fallback order
    Writes result to st.session_state['job_order'].
    """
    if default_order is None:
        default_order = DEFAULT_JOB_ORDER

    st.subheader("⚙️ Job Level Sequencing")

    # Derive job levels from emp_df if present
    job_levels = default_order.copy()
    if emp_df is not None and "JobLevel" in emp_df.columns:
        # keep ordering stable - unique sorted by appearance in default + found new ones appended
        found = list(emp_df["JobLevel"].dropna().unique())
        # order: keep default order first then append any new levels preserving order in found
        job_levels = [lvl for lvl in default_order if lvl in found] + [lvl for lvl in found if lvl not in default_order]
        if not job_levels:
            job_levels = default_order.copy()

    # initialize
    if "job_order" not in st.session_state:
        st.session_state.job_order = job_levels

    st.markdown(
        "Assign rank/order to job levels. Click **Apply Order** to persist selection across the consolidated module."
    )

    # Render selectboxes in a compact fashion
    ranked_levels = {}
    cols = st.columns(2)  # two columns for compactness
    for i, level in enumerate(job_levels):
        with cols[i % 2]:
            idx = job_levels.index(level) if level in job_levels else 0
            rank = st.selectbox(
                f"{level}",
                options=list(range(1, len(job_levels) + 1)),
                index=idx,
                key=f"_seq_{level}"
            )
            ranked_levels[level] = rank

    # Apply / Restore buttons
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Apply Order", use_container_width=True):
            ordered = [lvl for lvl, r in sorted(ranked_levels.items(), key=lambda x: x[1])]
            st.session_state.job_order = ordered
            st.success(f"Updated hierarchy: {', '.join(ordered)}")
            save_session_state()
    with c2:
        if st.button("↩️ Restore Default Order", use_container_width=True):
            st.session_state.job_order = default_order
            st.success("Restored default order.")
            save_session_state()

    # display active order
    st.info(f"Current hierarchy: {', '.join(st.session_state.get('job_order', default_order))}")


def _ensure_joblevel_order(df: pd.DataFrame, col="JobLevel"):
    """Applies the selected hierarchy order globally on a copy of df."""
    if not isinstance(df, pd.DataFrame):
        return df
    order = st.session_state.get("job_order", DEFAULT_JOB_ORDER)
    if col in df.columns:
        df = df.copy()
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df

# ---------------------------
# Chatbot (single bot for all modules)
# ---------------------------
def _sanitize_prompt_for_filters(prompt: str):
    """Split prompt into tokens; quick normalizations for matching against values."""
    return [p.strip() for p in prompt.lower().split() if p.strip()]


def run_chatbot_ui(modules_data: dict[str, pd.DataFrame] | None = None, primary_table_key: str | None = None):
    """
    Run chatbot UI. 
    - modules_data: dict mapping module_name -> DataFrame (e.g., {"compensation": emp_df, "benchmark": bench_df, "attrition": attr_df})
    - primary_table_key: prefer this as the default 'primary' dataframe name (like 'compensation' or 'emp')
    Usage: call from consolidated page. Ensure you have uploaded/loaded dataframes before enabling bot.
    """
    st.subheader("💬 Global HR Chatbot — Smart Assistant")

    # initialize messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Show history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # chat input
    prompt = st.chat_input("Ask anything about CTC, Bonus, Gender gap, Market, Attrition, or Job Level sequencing...")
    if not prompt:
        return

    # save user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    # autosave messages right away
    auto_save_session_state()

    q = prompt.lower()

    # choose primary df
    df = None
    if modules_data:
        if primary_table_key and primary_table_key in modules_data:
            df = modules_data.get(primary_table_key)
        else:
            # prefer common keys
            for k in ("compensation", "emp", "employee", "employees"):
                if k in modules_data:
                    df = modules_data[k]; break
            if df is None:
                # pick the largest dataframe present
                df = max(modules_data.values(), key=lambda d: d.shape[0]) if len(modules_data) else None

    # quick fallback message
    fallback = "🤔 I didn't catch that. Try asking: 'Average CTC by Job Level', 'Bonus % for Directors in Finance', 'Gender gap by level', 'Compare Company vs Market'."

    # defensive: empty df fallback
    if df is None or (hasattr(df, "empty") and df.empty):
        res = "No data loaded for the chatbot. Upload or pass module dataframes to the consolidated module for the bot to operate."
        st.session_state["messages"].append({"role": "assistant", "content": res})
        with st.chat_message("assistant"):
            st.markdown(res)
        return

    # normalize columns
    df = df.copy()
    col_lower_map = {c.lower(): c for c in df.columns}
    # provide handy column mapping
    def col(name):
        return col_lower_map.get(name.lower())

    # basic enrichments
    if col("ctc"):
        df["CTC"] = pd.to_numeric(df[col("ctc")], errors="coerce") if col("ctc") else df.get("CTC", None)
    if col("bonus") and "Bonus" not in df.columns:
        df["Bonus"] = pd.to_numeric(df[col("bonus")], errors="coerce")
    # compute bonus% if possible
    if "CTC" in df.columns and "Bonus" in df.columns:
        df["Bonus %"] = np.where(df["CTC"] > 0, (df["Bonus"] / df["CTC"]) * 100, np.nan)

    # detect intents (simple keyword checks)
    intents = []
    if any(k in q for k in ["bonus", "bonus %", "bonus%"]):
        intents.append("bonus")
    if any(k in q for k in ["avg", "average", "mean", "ctc"]):
        intents.append("average_ctc")
    if any(k in q for k in ["median", "median ctc"]):
        intents.append("median_ctc")
    if "gender" in q or "pay gap" in q:
        intents.append("gender_gap")
    if "market" in q or "compare" in q:
        intents.append("market_compare")
    if "attrition" in q or "leave" in q or "turnover" in q:
        intents.append("attrition")
    if "job level" in q or "joblevel" in q or "hierarchy" in q:
        intents.append("job_level")
    if not intents:
        intents = ["average_ctc"]  # default safe

    # extract simple filters (multi-value)
    filters = {}
    for field in ("JobLevel", "Department", "Gender"):
        if field in df.columns:
            values = df[field].dropna().astype(str).unique().tolist()
            matches = [v for v in values if v.lower() in q]
            if matches:
                filters[field] = matches

    # apply filters
    if filters:
        for k, vals in filters.items():
            df = df[df[k].isin(vals)]

    # produce results for each detected intent (first match is primary)
    assistant_texts = []
    assistant_chart = None
    primary_intent = intents[0]

    try:
        if primary_intent == "average_ctc":
            if "JobLevel" in df.columns and "CTC" in df.columns:
                agg = df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index()
                agg["CTC (₹ Lakhs)"] = (agg["CTC"] / 1e5).round(2)
                table_md = agg[['JobLevel', 'CTC (₹ Lakhs)']].to_markdown(index=False)
                assistant_texts.append(f"📊 **Average CTC by Job Level**\n\n{table_md}")
                assistant_chart = px.bar(agg, x="JobLevel", y="CTC (₹ Lakhs)", color="JobLevel", text="CTC (₹ Lakhs)")
        elif primary_intent == "median_ctc":
            if "JobLevel" in df.columns and "CTC" in df.columns:
                med = df.groupby("JobLevel", observed=True)["CTC"].median().reset_index()
                med["CTC (₹ Lakhs)"] = (med["CTC"] / 1e5).round(2)
                assistant_texts.append("📏 **Median CTC by Job Level**\n\n" + med[['JobLevel','CTC (₹ Lakhs)']].to_markdown(index=False))
                assistant_chart = px.bar(med, x="JobLevel", y="CTC (₹ Lakhs)", color="JobLevel", text="CTC (₹ Lakhs)")
        elif primary_intent == "bonus":
            if "JobLevel" in df.columns and "Bonus %" in df.columns:
                b = df.groupby("JobLevel", observed=True)["Bonus %"].mean().reset_index().round(2)
                assistant_texts.append("🎁 **Bonus % by Job Level**\n\n" + b.to_markdown(index=False))
                assistant_chart = px.bar(b, x="JobLevel", y="Bonus %", color="JobLevel", text="Bonus %")
            else:
                assistant_texts.append("Bonus % not found — ensure Compensation data includes Bonus and CTC columns.")
        elif primary_intent == "gender_gap":
            if "JobLevel" in df.columns and "Gender" in df.columns and "CTC" in df.columns:
                g = df.groupby(["JobLevel", "Gender"], observed=True)["CTC"].mean().reset_index()
                g["CTC (₹ Lakhs)"] = (g["CTC"] / 1e5).round(2)
                assistant_texts.append("👫 **Gender Pay by Job Level**\n\n" + g.pivot(index='JobLevel', columns='Gender', values='CTC (₹ Lakhs)').to_markdown())
                assistant_chart = px.bar(g, x="JobLevel", y="CTC (₹ Lakhs)", color="Gender", barmode="group")
            else:
                assistant_texts.append("Gender gap data requires 'Gender' and 'CTC' columns.")
        elif primary_intent == "market_compare":
            # requires that modules_data contains a benchmark table with "MarketMedianCTC" or similar
            if modules_data:
                bench_candidates = [k for k, d in modules_data.items() if any("market" in c.lower() for c in d.columns)]
                if bench_candidates:
                    bench = modules_data[bench_candidates[0]]
                    # join medians by job level (if present)
                    if "JobLevel" in df.columns and ("MarketMedianCTC" in bench.columns or "MarketMedian" in bench.columns):
                        left = df.groupby("JobLevel", observed=True)["CTC"].median().reset_index().rename(columns={"CTC":"CompanyMedian"})
                        right_col = "MarketMedianCTC" if "MarketMedianCTC" in bench.columns else ("MarketMedian" if "MarketMedian" in bench.columns else None)
                        right = bench.groupby("JobLevel", observed=True)[right_col].median().reset_index().rename(columns={right_col:"MarketMedian"})
                        cmp = pd.merge(left, right, on="JobLevel", how="inner").dropna()
                        if not cmp.empty:
                            cmp["Company (₹ L)"] = (cmp["CompanyMedian"] / 1e5).round(2)
                            cmp["Market (₹ L)"] = (cmp["MarketMedian"] / 1e5).round(2)
                            assistant_texts.append("📉 **Company vs Market Median**\n\n" + cmp[['JobLevel','Company (₹ L)','Market (₹ L)']].to_markdown(index=False))
                            assistant_chart = px.line(cmp, x="JobLevel", y=["Company (₹ L)", "Market (₹ L)"], markers=True)
                        else:
                            assistant_texts.append("No overlapping JobLevel rows found between company and benchmark data.")
                    else:
                        assistant_texts.append("Benchmark or JobLevel data not available for market comparison.")
                else:
                    assistant_texts.append("No benchmark dataset found in modules for market comparison.")
            else:
                assistant_texts.append("No module datasets available for market comparison.")
        elif primary_intent == "attrition":
            if modules_data and "attrition" in modules_data:
                a = modules_data["attrition"]
                if "AttritionFlag" in a.columns and "JobLevel" in a.columns:
                    rates = (a.groupby("JobLevel", observed=True)["AttritionFlag"]
                             .apply(lambda s: (s.astype(str).str.lower() == "yes").sum())
                             .reset_index(name="Left"))
                    counts = a.groupby("JobLevel", observed=True)["AttritionFlag"].count().reset_index(name="Total")
                    r = pd.merge(rates, counts, on="JobLevel")
                    r["Attrition %"] = (r["Left"] / r["Total"] * 100).round(1)
                    assistant_texts.append("📉 **Attrition % by Job Level**\n\n" + r[['JobLevel','Attrition %']].to_markdown(index=False))
                    assistant_chart = px.bar(r, x="JobLevel", y="Attrition %", color="JobLevel", text="Attrition %")
                else:
                    assistant_texts.append("Attrition table missing JobLevel / AttritionFlag columns.")
            else:
                assistant_texts.append("No attrition dataset supplied.")
        elif primary_intent == "job_level":
            # return current sequence if available
            seq = st.session_state.get("job_order", DEFAULT_JOB_ORDER)
            assistant_texts.append(f"🧭 Current Job Level hierarchy:\n\n{', '.join(seq)}")
        else:
            assistant_texts.append(fallback)
    except Exception as e:
        assistant_texts.append(f"⚠️ Error while computing result: {e}")

    # format assistant response
    final_resp = "\n\n".join(assistant_texts) if assistant_texts else fallback

    st.session_state["messages"].append({"role": "assistant", "content": final_resp})
    auto_save_session_state()  # quick save

    # render assistant message
    with st.chat_message("assistant"):
        st.markdown(final_resp)
        if assistant_chart is not None:
            assistant_chart = _apply_small_chart_style(assistant_chart)
            st.plotly_chart(assistant_chart, use_container_width=True)


def _apply_small_chart_style(fig):
    """Small internal helper to make charts readable from chatbot (no colors forced)."""
    try:
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), height=420)
        return fig
    except Exception:
        return fig

# ---------------------------
# Small helper: call at module import from app/consolidated
# ---------------------------
def bootstrap_persistence():
    """Call early in consolidated page to auto-load and ensure keys exist."""
    # ensure session keys exist
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "job_order" not in st.session_state:
        st.session_state["job_order"] = DEFAULT_JOB_ORDER.copy()
    # attempt to auto-load previous session quietly
    auto_load_session_state()
    # run an autosave to create file if keys present
    auto_save_session_state()

# End of utils/persistence_chatbot.py