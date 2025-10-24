# ============================================
# pages/consolidated.py — v4.0 | Executive Stable (Post-Integration)
# ============================================
"""
📘 Consolidated HR Leadership Deck Entry Point
------------------------------------------------
Loads and displays the unified executive dashboard
that merges module reports from:
Workforce, Performance, Engagement, Compensation, Attrition

✅ Reflects real-time deck status (from TMP_DIR)
✅ Uses deck_state_tracker timestamps
✅ Allows single-click final PDF merge
✅ Keeps global styling consistent
"""

import streamlit as st
import os
from utils_consolidated.deck_state_tracker import get_module_state
from utils_consolidated.pdf_merger import TMP_DIR, merge_pdfs

# -------------------------------------------------------
# 🧭 Page Identity
# -------------------------------------------------------
st.set_page_config(
    page_title="Consolidated HR Leadership Deck",
    page_icon="📘",
    layout="wide"
)

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
# 🧾 Merge Final Deck
# -------------------------------------------------------
st.markdown("---")
st.header("📄 Generate Consolidated Executive Report")

st.caption("Combines all completed module PDFs into a single master HR Leadership Deck.")

if st.button("🧾 Merge & Generate Consolidated PDF", use_container_width=True):
    output_path = os.path.join(TMP_DIR, "People_Analytics_Leadership_Deck.pdf")
    try:
        success = merge_pdfs(output_path)
        if success and os.path.exists(output_path):
            st.success("✅ Consolidated Leadership Deck generated successfully!")
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download HR Leadership Deck (PDF)",
                    f,
                    file_name="People_Analytics_Leadership_Deck.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("⚠️ Some module PDFs are missing. Add them before merging.")
    except Exception as e:
        st.error(f"❌ Failed to merge PDFs: {e}")

# -------------------------------------------------------
# 📁 Folder path helper (for debugging)
# -------------------------------------------------------
with st.expander("📂 View Consolidation Folder"):
    st.write(f"**TMP_DIR:** `{TMP_DIR}`")
    st.write(pdf_files)