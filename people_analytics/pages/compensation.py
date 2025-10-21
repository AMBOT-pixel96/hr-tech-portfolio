import streamlit as st
st.set_page_config(page_title="Compensation Analytics", layout="wide")

from utils.ui_styling import apply_sidebar_theme
from modules.compensation_module import run_compensation_module

apply_sidebar_theme()
run_compensation_module()