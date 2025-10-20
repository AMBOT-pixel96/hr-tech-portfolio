# pages/attrition.py
import streamlit as st
from modules.attrition_module import run_attrition_module
from utils.template_helper import render_download_template

st.set_page_config(page_title="Attrition Analytics", layout="wide")
run_attrition_module()