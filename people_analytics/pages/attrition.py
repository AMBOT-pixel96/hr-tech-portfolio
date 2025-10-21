import streamlit as st
st.set_page_config(page_title="Attrition Analytics", layout="wide")

from utils.ui_styling import apply_sidebar_theme
from modules.attrition_module import run_attrition_module

apply_sidebar_theme()
run_attrition_module()