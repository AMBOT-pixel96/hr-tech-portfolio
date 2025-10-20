# pages/workforce.py
import streamlit as st
from modules.workforce_module import run_workforce_module
from utils.template_helper import render_download_template

st.set_page_config(page_title="Workforce & Talent Analytics", layout="wide")
run_workforce_module()