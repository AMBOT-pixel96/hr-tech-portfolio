import streamlit as st
import pandas as pd

# --- Unified Styling for All Download Buttons ---
st.markdown("""
<style>
.stDownloadButton button {
    background: linear-gradient(90deg, #1E3A8A, #3B82F6);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
}
.stDownloadButton button:hover {
    background: linear-gradient(90deg, #2563EB, #60A5FA);
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)
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