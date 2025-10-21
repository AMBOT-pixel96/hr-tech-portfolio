# ============================================
# utils/ui_styling.py — v1.0 | Unified Sidebar & Theme Styling
# ============================================

import streamlit as st

def apply_sidebar_theme():
    """
    Re-applies the Executive Edition sidebar and dark theme styles
    across all Streamlit pages/modules.
    Call this function at the bottom of each page (e.g., pages/performance.py).
    """
    st.markdown("""
    <style>
    /* Sidebar gradient and text */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E3A8A 100%);
        color: white;
        padding-top: 1rem;
        border-right: 1px solid #1E293B;
    }

    /* Dashboard Title */
    [data-testid="stSidebarNav"]::before {
        content: "📊 People Analytics Dashboard";
        margin-left: 20px;
        font-weight: 700;
        font-size: 18px;
        color: #FACC15;
        text-transform: uppercase;
    }

    /* Sidebar links */
    [data-testid="stSidebarNav"] a {
        color: #E2E8F0 !important;
        font-weight: 500;
        border-radius: 8px;
        padding: 10px 15px;
        transition: all 0.2s ease-in-out;
        text-transform: capitalize;
    }

    [data-testid="stSidebarNav"] a:hover {
        background: rgba(255,255,255,0.1);
        transform: scale(1.03);
    }

    [data-testid="stSidebarNav"] a span::before {
        margin-right: 8px;
    }

    /* Icons for each module */
    a[href*="performance"] span::before { content: "🏆 "; }
    a[href*="engagement"] span::before { content: "💬 "; }
    a[href*="compensation"] span::before { content: "💰 "; }
    a[href*="attrition"] span::before { content: "📉 "; }
    a[href*="workforce"] span::before { content: "🏢 "; }
    a[href*="app"] span::before { content: "🏠 "; }

    /* Active Link Highlight */
    [data-testid="stSidebarNav"] a[data-testid="stSidebarNavLinkActive"] {
        background: #1D4ED8;
        color: white !important;
        font-weight: 700;
    }

    /* Hover glow for the title */
    [data-testid="stSidebarNav"]::before:hover {
        text-shadow: 0px 0px 8px #FACC15;
        transition: 0.3s ease-in-out;
    }

    /* Global body styling for dark mode */
    body {
        background-color: #0E1117;
        color: white;
    }
    h1, h2, h3, h4 {
        color: #F9FAFB;
    }
    </style>
    """, unsafe_allow_html=True)

    st.caption("🎨 Executive theme applied successfully.")