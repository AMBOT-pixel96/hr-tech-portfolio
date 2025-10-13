import streamlit as st
import pandas as pd

def render_download_template(title: str, sample_data: pd.DataFrame, filename: str):
    """
    Reusable Download Template Section for all modules.
    
    Args:
        title (str): Title to display (e.g., "Performance Data Template")
        sample_data (pd.DataFrame): Sample data to generate CSV
        filename (str): Name of the downloadable file
    """
    st.subheader(f"📥 Download {title}")
    
    csv_data = sample_data.to_csv(index=False)
    st.download_button(
        label=f"⬇️ Download {title}",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )
    st.caption("Use this format to prepare your input file for uploading.")