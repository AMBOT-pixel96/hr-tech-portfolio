# ============================================
# pages/consolidated.py — v4.4 | Kage Stable Edition (Sidebar + UX Fix)
# ============================================
"""
📘 Consolidated HR Leadership Deck
------------------------------------------------
Combines all module PDFs (Workforce, Performance,
Engagement, Compensation, Attrition) into one unified
executive deck for boardroom presentation.

✅ Real-time deck status with timestamps
✅ Unified sidebar theme
✅ Single-click final merge
✅ Consistent visual language across app
"""

# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import os, sys
import streamlit as st
from utils_consolidated.pdf_merger import TMP_DIR, merge_pdfs
from utils_consolidated.deck_state_tracker import get_module_state

# -------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Consolidated HR Leadership Deck",
    page_icon="📘",
    layout="wide"
)

# -------------------------------------------------------
# 🎨 Global Sidebar Styling Fix — Consistent Across Pages
# -------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E3A8A 100%) !important;
    color: white !important;
    padding-top: 1rem !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebarNav"]::before {
    content: "📘 CONSOLIDATED HR LEADERSHIP DECK";
    margin-left: 16px;
    font-weight: 800;
    font-size: 15px;
    color: #FACC15;
    text-transform: uppercase;
}
[data-testid="stSidebarNav"] a {
    color: #E5E7EB !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    margin: 2px 8px !important;
    padding: 4px 10px !important;
}
[data-testid="stSidebarNav"] a:hover {
    background-color: rgba(255,255,255,0.1) !important;
    color: #FACC15 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 🎨 Header Banner
# -------------------------------------------------------
st.markdown("""
<div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
  <h2 style="margin:0;">📘 Consolidated HR Leadership Deck</h2>
  <p style="margin:4px 0 0 0;">Unified report combining all module PDFs into a single boardroom-ready document.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 🧩 Deck Queue Visualizer
# -------------------------------------------------------
st.markdown("### 🧩 Current Deck Status")

os.makedirs(TMP_DIR, exist_ok=True)
pdf_files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
modules_expected = ["Workforce", "Performance", "Engagement", "Compensation", "Attrition"]
state = get_module_state()

cols = st.columns(len(modules_expected))
for i, mod in enumerate(modules_expected):
    icon = "✅" if f"{mod}.pdf" in pdf_files else "❌"
    last_updated = state.get(mod, "—")
    with cols[i]:
        st.markdown(f"""
        <div style="padding:12px;border-radius:10px;background:rgba(255,255,255,0.05);
                    border:1px solid #1E3A8A;text-align:center;">
          <h4 style="margin:0;color:#FACC15;">{icon} {mod}</h4>
          <p style="margin:2px 0 0;color:#E5E7EB;font-size:13px;">
            {'Added to Deck' if icon=='✅' else 'Pending'}
          </p>
          <p style="margin:0;color:#9CA3AF;font-size:11px;">🕒 {last_updated.split('T')[0] if last_updated!='—' else '—'}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# 🧾 Generate Final Deck
# -------------------------------------------------------
st.header("📄 Finalize & Generate Executive Leadership Deck")
st.caption("Combines all added module PDFs into one executive-ready leadership report.")

if st.button("🧾 Merge & Generate Consolidated Deck", use_container_width=True):
    try:
        merge_pdfs()
    except Exception as e:
        st.error(f"❌ Failed to merge PDFs: {e}")

# -------------------------------------------------------
# 🧹 Maintenance Tools
# -------------------------------------------------------
st.markdown("---")
st.markdown("### 🧹 Maintenance Options")

col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Deck Queue", use_container_width=True):
        try:
            for f in os.listdir(TMP_DIR):
                os.remove(os.path.join(TMP_DIR, f))
            st.success("✅ Cleared all queued PDFs successfully.")
        except Exception as e:
            st.error(f"⚠️ Failed to clear queue: {e}")

with col2:
    if st.button("📂 Show Files in Deck Folder", use_container_width=True):
        files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
        if not files:
            st.info("No PDFs currently in queue.")
        else:
            st.write("**Queued PDFs:**")
            for f in files:
                st.write(f"📄 {f}")