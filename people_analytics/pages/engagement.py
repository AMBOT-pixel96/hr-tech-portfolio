# pages/engagement.py
import streamlit as st
from modules.engagement_module import run_engagement_module

# Optional: hide default Streamlit page header for cleaner look
st.set_page_config(page_title="Engagement", layout="wide")
run_engagement_module()