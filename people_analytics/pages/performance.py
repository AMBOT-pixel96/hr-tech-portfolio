import streamlit as st
from modules.performance_module import run_performance_module

# --- Sidebar Customization ---
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        color: #FACC15;
        text-align: center;
        margin-bottom: 10px;
    }
    .sidebar-button {
        display: block;
        width: 100%;
        padding: 10px 15px;
        background: linear-gradient(90deg,#1E3A8A,#3B82F6);
        color: white;
        border: none;
        border-radius: 8px;
        text-align: center;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .sidebar-button:hover {
        background: linear-gradient(90deg,#2563EB,#60A5FA);
        transform: scale(1.03);
    }
    </style>

    <p class="sidebar-title">🏆 Performance Module</p>
    <a class="sidebar-button" href="#" target="_self">Upload Data</a>
    <a class="sidebar-button" href="#" target="_self">Metrics View</a>
    <a class="sidebar-button" href="#" target="_self">Export & Reports</a>
    """, unsafe_allow_html=True)

# --- Run the Module ---
run_performance_module()
# --- End --- #