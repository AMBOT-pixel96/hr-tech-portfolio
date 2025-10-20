# ============================================
# utils/pdf_auto_exporter.py — Dynamic Data → PDF Builder
# ============================================

import pandas as pd
import streamlit as st
from utils.pdf_helper import render_pdf_download_button

# ==================================================
# Helper — Build Executive Report Sections Automatically
# ==================================================
def build_report_sections(data_blocks):
    """
    Converts Streamlit module metrics into standardized PDF sections.

    Args:
        data_blocks (list): A list of dictionaries, each like:
            {
                "title": "Metric Title",
                "desc": "Brief description",
                "df": pandas.DataFrame or list of lists,
                "insights": ["point 1", "point 2"]
            }

    Returns:
        (sections, insights_dict)
    """
    sections = []
    insights_dict = {}

    for block in data_blocks:
        title = block.get("title", "Untitled Metric")
        desc = block.get("desc", "")
        df = block.get("df")
        insights = block.get("insights", [])

        # Convert DataFrame to nested list if needed
        if isinstance(df, pd.DataFrame):
            table_data = [df.columns.tolist()] + df.astype(str).values.tolist()
        elif isinstance(df, list):
            table_data = df
        else:
            table_data = None

        sections.append({
            "title": title,
            "desc": desc,
            "table": table_data,
            "insights": insights
        })

        if insights:
            insights_dict[title] = " | ".join(insights)

    return sections, insights_dict


# ==================================================
# Streamlit Wrapper — Universal PDF Exporter
# ==================================================
def export_module_report(
    report_title: str,
    module_name: str,
    data_blocks: list,
    filename_prefix: str
):
    """
    Generates a PDF from the provided data blocks dynamically.

    Example usage inside a module:
    --------------------------------
    data_blocks = [
        {
            "title": "Performance Distribution",
            "desc": "Employee performance bell curve",
            "df": df_perf_summary,
            "insights": [
                "Majority employees rated 4 or above",
                "Balanced spread across departments"
            ]
        },
        {
            "title": "Performance vs Pay",
            "desc": "CTC distribution by rating level",
            "df": df_ctc_summary,
            "insights": ["High correlation between rating and pay"]
        }
    ]
    export_module_report("Performance Analytics Report", "Performance", data_blocks, "Performance")
    """
    try:
        st.subheader("📄 Export Executive Report (PDF)")
        sections, insights_dict = build_report_sections(data_blocks)
        render_pdf_download_button(
            report_title=report_title,
            module_name=module_name,
            sections=sections,
            all_insights=insights_dict,
            filename_prefix=filename_prefix
        )
    except Exception as e:
        st.error(f"⚠️ Error while exporting PDF: {e}")