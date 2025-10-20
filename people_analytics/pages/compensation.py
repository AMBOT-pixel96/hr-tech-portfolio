# pages/compensation.py
import streamlit as st
from modules.compensation_module import run_compensation_module
from utils.template_helper import render_download_template

st.set_page_config(page_title="Compensation Analytics", layout="wide")
run_compensation_module()