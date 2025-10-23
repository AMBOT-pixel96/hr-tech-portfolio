# ============================================
# pages/consolidated.py — v5.0 | Unified HR Leadership Deck Entry
# ============================================
"""
Streamlit Page Entry for the Consolidated HR Executive Deck.
This page imports and runs the consolidated module that
combines all functional datasets into one branded PDF report.
"""

import streamlit as st
from modules.consolidated_module import run_consolidated_module


def main():
    # === Header Banner ===
    st.markdown("""
    <div style="
        padding:18px;
        border-radius:10px;
        background:linear-gradient(90deg,#111827,#0F172A);
        color:white;">
        <h2 style="margin:0;">🏢 Consolidated HR Leadership Deck</h2>
        <p style="margin:4px 0 0 0;font-size:15px;">
            One unified report bringing together insights across Attrition, Compensation, Workforce, Engagement, and Performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Overview")
    st.write("""
    The **Consolidated HR Leadership Deck** merges all individual HR analytics modules into
    a single, boardroom-ready executive report.  
    You can upload datasets from each functional area (Attrition, Compensation, Workforce, Engagement, Performance)
    and generate a unified PDF containing:
    
    - 📘 Cover Page & TOC  
    - 📊 Department-wise Metrics & Charts  
    - 🧩 Sectional Summaries for Each Function  
    - 🧾 Executive Summary (Key Insights)
    
    ---
    """)

    # === Launch consolidated module ===
    run_consolidated_module()

    # === Footer ===
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#6B7280;font-size:13px;margin-top:8px;">
        Built with ❤️ using Streamlit, Plotly, and ReportLab · © 2025 People Analytics Project
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()