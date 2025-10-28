# ============================================
# pages/consolidated.py — v6.7 | Persistence + Chatbot Cross-Module Intelligence Arc
# ============================================
"""
📘 Consolidated HR Leadership Deck Entry Point
------------------------------------------------
Loads and displays the unified executive dashboard
that merges module reports from:
Workforce, Performance, Engagement, Compensation, Attrition

✅ Case-insensitive PDF validation
✅ Reflects real-time deck status (from TMP_DIR)
✅ Uses deck_state_tracker timestamps
✅ Allows single-click final PDF merge
✅ Includes Maintenance Panel
✅ Uniform Sidebar Styling
✅ Job-Level Sequencer (Persistent)
✅ Global Chatbot (Cross-Module Intelligence)
✅ Auto-loads module dataframes from session memory
"""

import sys, os
import streamlit as st
import pandas as pd
from datetime import datetime

# -------------------------------------------------------
# 🧭 Import setup
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from utils_consolidated.pdf_merger import TMP_DIR, merge_consolidated_pdfs
    from utils_consolidated.deck_state_tracker import get_module_state
except ModuleNotFoundError as e:
    st.error(f"⚠️ Import error: {e}")
    st.stop()

# -------------------------------------------------------
# ⚙️ Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Consolidated HR Leadership Deck",
    page_icon="📘",
    layout="wide"
)

# -------------------------------------------------------
# 🧠 Import Persistence + Chatbot Utilities
# -------------------------------------------------------
from utils.persistence_chatbot import (
    bootstrap_persistence,
    job_level_sequencer_ui,
    run_chatbot_ui
)

# -------------------------------------------------------
# 🧠 Initialize Session (Memory Restore)
# -------------------------------------------------------
bootstrap_persistence()

# -------------------------------------------------------
# 🎨 Executive Styling
# -------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E3A8A 100%);
    color: white;
    padding-top: 1rem;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebarNav"]::before {
    content: "📘 CONSOLIDATED HR LEADERSHIP DECK";
    margin-left: 20px;
    font-weight: 800;
    font-size: 15px;
    color: #FACC15;
    text-transform: uppercase;
}
h1, h2, h3, h4 {
    color: #F9FAFB;
}
.deck-status {
    border: 1px solid #1E3A8A;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
    text-align: center;
    padding: 10px;
    color: #E5E7EB;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 🏷️ Header
# -------------------------------------------------------
st.markdown("""
<div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
  <h2 style="margin:0;">📘 Consolidated HR Leadership Deck</h2>
  <p style="margin:4px 0 0 0;">Unified report combining all module PDFs into a single boardroom-ready document.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 🧩 Deck Status Tracker
# -------------------------------------------------------
st.markdown("### 🧩 Current Deck Status")

modules_expected = ["Workforce", "Performance", "Engagement", "Compensation", "Attrition"]
pdf_files = os.listdir(TMP_DIR) if os.path.exists(TMP_DIR) else []
state = get_module_state()

cols = st.columns(len(modules_expected))
for i, mod in enumerate(modules_expected):
    icon = "✅" if f"{mod}.pdf" in pdf_files else "❌"
    last_updated = state.get(mod, "—")
    with cols[i]:
        st.markdown(f"""
        <div class="deck-status">
            <h4 style="margin:0;color:#FACC15;">{icon} {mod}</h4>
            <p style="margin:2px 0 0;">{'Added to Deck' if icon=='✅' else 'Pending'}</p>
            <p style="margin:0;color:#9CA3AF;font-size:11px;">🕒 {last_updated.split('T')[0] if last_updated!='—' else '—'}</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------
# ⚙️ Job Level Sequencing (Persistent)
# -------------------------------------------------------
st.markdown("---")
st.header("⚙️ Job Level Hierarchy Sequencing (Persistent)")

# Use real data if loaded from other modules; fallback to dummy
emp_df = None
if "compensation_df" in st.session_state:
    emp_df = st.session_state["compensation_df"]
elif "emp_df" in st.session_state:
    emp_df = st.session_state["emp_df"]
else:
    emp_df = pd.DataFrame({
        "EmployeeID": [101, 102, 103, 104],
        "JobLevel": ["Analyst", "Manager", "Senior Manager", "Director"]
    })

job_level_sequencer_ui(emp_df=emp_df)

# -------------------------------------------------------
# 🧾 Merge Final Deck (with case-insensitive validation)
# -------------------------------------------------------
st.markdown("---")
st.header("📄 Finalize & Generate Executive Leadership Deck")
st.caption("Combines all completed module PDFs into a single master HR Leadership Deck.")

# Case-insensitive check
existing_pdfs = [os.path.splitext(f)[0].lower() for f in pdf_files if f.endswith(".pdf")]
missing = [m for m in modules_expected if m.lower() not in existing_pdfs]

if missing:
    st.warning(f"⚠️ Some module PDFs are missing: {', '.join(missing)}. Add them before merging.")
else:
    if st.button("🧾 Merge & Generate Consolidated Deck", use_container_width=True):
        output_path = os.path.join(TMP_DIR, "People_Analytics_Leadership_Deck.pdf")
        try:
            success = merge_consolidated_pdfs(output_path)
            if success and os.path.exists(output_path):
                st.success("✅ Consolidated Leadership Deck generated successfully!")
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Final Consolidated Deck",
                        f,
                        file_name="People_Analytics_Leadership_Deck.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("⚠️ Some module PDFs might still be missing. Please verify.")
        except Exception as e:
            st.error(f"❌ Failed to merge PDFs: {e}")

# -------------------------------------------------------
# 🧹 Maintenance Tools
# -------------------------------------------------------
st.markdown("---")
st.subheader("🧹 Maintenance Options")

col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Deck Queue", use_container_width=True):
        try:
            for f in os.listdir(TMP_DIR):
                os.remove(os.path.join(TMP_DIR, f))
            st.success("✅ Cleared all queued PDFs successfully.")
        except Exception as e:
            st.error(f"⚠️ Failed to clear: {e}")

with col2:
    with st.expander("📂 Show Files in Deck Folder"):
        files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf") or f.endswith(".json")]
        if not files:
            st.info("No files currently in the queue.")
        else:
            for f in files:
                path = os.path.join(TMP_DIR, f)
                mod_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%b'%y %H:%M")
                st.write(f"📄 {f} — {mod_time}")

# -------------------------------------------------------
# 🤖 Sidebar Chatbot (Cross-Module Intelligence)
# -------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Smart HR Chatbot")

# 🧩 Try to automatically fetch module dataframes if they exist in session
modules = {}
for key in ["compensation_df", "attrition_df", "engagement_df", "performance_df", "workforce_df"]:
    if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
        short_key = key.replace("_df", "")
        modules[short_key] = st.session_state[key]

# Fallback: ensure at least one dataset exists
if not modules:
    modules["compensation"] = emp_df

# ✅ Run chatbot via sidebar toggle
if st.sidebar.checkbox("Enable Chatbot (Smart Mode)", value=False):
    run_chatbot_ui(modules_data=modules, primary_table_key="compensation")