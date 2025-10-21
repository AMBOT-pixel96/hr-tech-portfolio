import streamlit as st
st.set_page_config(page_title="Workforce Analytics", layout="wide")

from utils.ui_styling import apply_sidebar_theme
from modules.workforce_module import run_workforce_module

apply_sidebar_theme()
run_workforce_module()