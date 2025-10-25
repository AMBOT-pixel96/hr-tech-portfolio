# ============================================
# pages/consolidated.py — v4.4 | Executive Stable (Sidebar Fixed)
# ============================================
import os
import sys
import streamlit as st

# -------------------------------------------------------
# 📦 Safe Imports — Local + Cloud
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from utils_consolidated.pdf_merger import TMP_DIR, merge_pdfs
    from utils_consolidated.deck_state_tracker import get_module_state
except ModuleNotFoundError as e:
    st.error(f"⚠️ Import error: {e}")
    st.stop()

# -------------------------------------------------------
# 🧭 Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Consolidated HR Leadership Deck",
    page_icon="📘",
    layout="wide"
)

# -------------------------------------------------------
# 🎨 Styling
# -------------------------------------------------------
st.markdown("""
<style>
/* Sidebar gradient and header */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E3A8A 100%);
    color: white;
    padding-top: 1rem;
    border-right: 1px solid #1E293B;
}

/* Sidebar Header Title */
[data-testid="stSidebarNav"]::before {
    content: "📘 CONSOLIDATED HR LEADERSHIP DECK";
    margin-left: 20px;
    font-weight: 800;
    font-size: 15px;
    color: #FACC15;
    text-transform: uppercase;
}

/* Capitalize Sidebar Page Links */
[data-testid="stSidebarNav"] a p {
    text-transform: capitalize !important;
}

/* Highlight active page */
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(250, 204, 21, 0.15) !important;
    color: #FACC15 !important;
    font-weight: 700 !important;
}

/* Deck Status Cards */
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
# 🧾 Merge Final Deck
# -------------------------------------------------------
st.markdown("---")
st.header("📄 Finalize & Generate Executive Leadership Deck")
st.caption("Combines all added module PDFs into one executive-ready leadership report.")

if st.button("🧾 Merge & Generate Consolidated Deck", use_container_width=True):
    merge_pdfs()

# -------------------------------------------------------
# 🧹 Maintenance Options
# -------------------------------------------------------
st.markdown("---")
st.header("🧹 Maintenance Options")

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
    if st.button("📂 Show Files in Deck Folder", use_container_width=True):
        files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
        if not files:
            st.info("No PDFs currently in the queue.")
        else:
            for f in files:
                st.write(f"📄 {f}")