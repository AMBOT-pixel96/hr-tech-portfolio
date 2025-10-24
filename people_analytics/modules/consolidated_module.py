# ============================================
# modules/consolidated_module.py — v6.0 | Boardroom Assembler Edition
# ============================================
import os
import streamlit as st
from datetime import datetime
from utils_consolidated.pdf_merger import merge_consolidated_pdfs, TMP_DIR

# -------------------------------------------------------
# 🎨 Header Banner
# -------------------------------------------------------
st.markdown("""
<div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#0F172A,#1E3A8A);color:white;">
  <h2 style="margin:0">📘 Consolidated HR Leadership Deck</h2>
  <p style="margin:4px 0 0 0;">Merge all module reports into one boardroom-ready PDF.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 📁 Directory Setup
# -------------------------------------------------------
os.makedirs(TMP_DIR, exist_ok=True)

# -------------------------------------------------------
# 🧩 Deck Queue Visualizer (new)
# -------------------------------------------------------
st.markdown("### 🧾 Current Deck Queue")
pdf_files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
modules_expected = ["Attrition", "Compensation", "Performance", "Engagement", "Workforce"]

if not pdf_files:
    st.info("No reports have been added yet. Generate and add module PDFs first.")
else:
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    for i, mod in enumerate(modules_expected):
        icon = "✅" if f"{mod}.pdf" in pdf_files else "❌"
        with cols[i]:
            st.markdown(f"""
            <div style="padding:12px;border-radius:10px;background:rgba(255,255,255,0.05);
                        border:1px solid #1E3A8A;text-align:center;">
              <h4 style="margin:0;color:#FACC15;">{icon} {mod}</h4>
              <p style="margin:4px 0 0;color:#E5E7EB;font-size:13px;">
                {'Added to Deck' if icon=='✅' else 'Pending'}
              </p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# 🧠 Info Summary
# -------------------------------------------------------
st.caption("""
🧩 Each module generates its own executive PDF.
Once all are added via their “➕ Add to Consolidated Deck” buttons, 
you can merge them here into a single, styled leadership deck.
""")

# -------------------------------------------------------
# 🧾 Generate Final Deck
# -------------------------------------------------------
st.markdown("### 📘 Build Final Consolidated Leadership Deck")
st.caption("Combines all added module PDFs into one executive-ready report with dividers and a cover page.")

if st.button("🧾 Generate Final Consolidated Deck", use_container_width=True):
    merge_consolidated_pdfs()

# -------------------------------------------------------
# 🧹 Optional Maintenance
# -------------------------------------------------------
st.markdown("---")
st.markdown("### 🧹 Maintenance Options")

colA, colB = st.columns(2)
with colA:
    if st.button("🧹 Clear Deck Queue", use_container_width=True):
        try:
            for f in os.listdir(TMP_DIR):
                os.remove(os.path.join(TMP_DIR, f))
            st.success("✅ Cleared all queued PDFs successfully.")
        except Exception as e:
            st.error(f"⚠️ Failed to clear: {e}")

with colB:
    if st.button("📂 Open Deck Folder (List PDFs)", use_container_width=True):
        files = [f for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
        if not files:
            st.info("No PDFs currently in the queue.")
        else:
            for f in files:
                st.write(f"📄 {f} — {datetime.fromtimestamp(os.path.getmtime(os.path.join(TMP_DIR, f))).strftime('%d %b %Y %H:%M')}")