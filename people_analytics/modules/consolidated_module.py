# ============================================
# modules/consolidated_module.py — v5.0 | Enterprise Deck Mode
# ============================================
"""
Combines data from all HR analytics modules into one unified executive deck.
Each uploaded dataset is processed and visualized briefly,
then exported to a beautifully formatted consolidated PDF
via utils_consolidated helpers.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# === Consolidated utils ===
from utils_consolidated.pdf_consolidated_helper import render_consolidated_pdf
from utils_consolidated.chart_consolidated_saver import ensure_chart_saved
from utils_consolidated.insights_helper import flatten_insights

# === Shared uploader ===
from utils.uploader_helper import upload_data


# ==========================================================
# ⚙️ MAIN FUNCTION
# ==========================================================
def run_consolidated_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;
                background:linear-gradient(90deg,#111827,#0F172A);
                color:white;">
      <h2 style="margin:0">🧩 Consolidated HR Executive Deck</h2>
      <p style="margin:4px 0 0 0;">Upload all module datasets to generate a unified leadership report.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📤 Upload Data Files")
    st.write("Upload each module's processed dataset below:")

    # Upload individual module datasets
    df_attr = upload_data("Attrition Data")
    df_comp = upload_data("Compensation Data")
    df_work = upload_data("Workforce Data")
    df_eng = upload_data("Engagement Data")
    df_perf = upload_data("Performance Data")

    # If none uploaded, stop
    if not any([df_attr, df_comp, df_work, df_eng, df_perf]):
        st.info("📂 Please upload at least one dataset to continue.")
        return

    # ==========================================================
    # 🧠 PROCESS EACH DATASET INTO A DATA BLOCK
    # ==========================================================
    data_blocks = []

    # --- Attrition Block ---
    if df_attr is not None and "AttritionFlag" in df_attr.columns:
        total = len(df_attr)
        left = (df_attr["AttritionFlag"].astype(str).str.lower().isin(["yes", "y", "1", "true"])).sum()
        rate = (left / total * 100) if total else 0
        avg_tenure = df_attr["TenureMonths"].mean() if "TenureMonths" in df_attr.columns else None

        dept = df_attr.groupby("Department", observed=True)["AttritionFlag"].apply(lambda x: (x.astype(str).str.lower().isin(["yes", "y", "1", "true"])).mean() * 100).reset_index(name="Rate")

        fig_attr = px.bar(dept, x="Department", y="Rate", text="Rate", color="Department",
                          title="Attrition by Department", color_discrete_sequence=px.colors.qualitative.Vivid)
        insights = [f"Attrition rate: {rate:.1f}%", f"Avg Tenure: {avg_tenure:.1f} months" if avg_tenure else ""]
        data_blocks.append({
            "title": "Attrition Analysis",
            "desc": "Attrition percentage and department-wise distribution.",
            "df": dept,
            "fig": fig_attr,
            "insights": insights
        })

    # --- Compensation Block ---
    if df_comp is not None and "Department" in df_comp.columns and "Salary" in df_comp.columns:
        comp_summary = df_comp.groupby("Department", observed=True)["Salary"].agg(["mean", "median", "max"]).reset_index()
        fig_comp = px.bar(comp_summary, x="Department", y="mean", text="mean", color="Department",
                          title="Average Salary by Department", color_discrete_sequence=px.colors.qualitative.Vivid)
        insights = [f"Overall average salary: ₹{df_comp['Salary'].mean():,.0f}"]
        data_blocks.append({
            "title": "Compensation Overview",
            "desc": "Average, median, and maximum salary across departments.",
            "df": comp_summary,
            "fig": fig_comp,
            "insights": insights
        })

    # --- Workforce Block ---
    if df_work is not None and "Department" in df_work.columns and "Headcount" in df_work.columns:
        fig_work = px.bar(df_work, x="Department", y="Headcount", text="Headcount", color="Department",
                          title="Department-wise Headcount", color_discrete_sequence=px.colors.qualitative.Vivid)
        insights = [f"Total workforce size: {df_work['Headcount'].sum()}"]
        data_blocks.append({
            "title": "Workforce Overview",
            "desc": "Headcount distribution by department.",
            "df": df_work,
            "fig": fig_work,
            "insights": insights
        })

    # --- Engagement Block ---
    if df_eng is not None and "EngagementScore" in df_eng.columns:
        eng_summary = df_eng.groupby("Department", observed=True)["EngagementScore"].mean().reset_index(name="AvgScore")
        fig_eng = px.bar(eng_summary, x="Department", y="AvgScore", text="AvgScore", color="Department",
                         title="Average Engagement Score by Department", color_discrete_sequence=px.colors.qualitative.Vivid)
        insights = [f"Overall engagement score: {df_eng['EngagementScore'].mean():.1f}/5"]
        data_blocks.append({
            "title": "Employee Engagement",
            "desc": "Average engagement scores across departments.",
            "df": eng_summary,
            "fig": fig_eng,
            "insights": insights
        })

    # --- Performance Block ---
    if df_perf is not None and "PerformanceRating" in df_perf.columns:
        perf_summary = df_perf.groupby("Department", observed=True)["PerformanceRating"].mean().reset_index(name="AvgRating")
        fig_perf = px.bar(perf_summary, x="Department", y="AvgRating", text="AvgRating", color="Department",
                          title="Average Performance Rating by Department", color_discrete_sequence=px.colors.qualitative.Vivid)
        insights = [f"Overall performance rating: {df_perf['PerformanceRating'].mean():.2f}/5"]
        data_blocks.append({
            "title": "Performance Overview",
            "desc": "Average performance ratings across departments.",
            "df": perf_summary,
            "fig": fig_perf,
            "insights": insights
        })

    # ==========================================================
    # 📊 FINAL RENDER SECTION
    # ==========================================================
    st.markdown("---")
    st.subheader("📘 Generate Leadership Deck")
    st.caption("Combines all module insights, charts, and summaries into a single boardroom-ready PDF.")
    render_consolidated_pdf(
        report_title="People Analytics – Leadership Insights Deck",
        module_label="Consolidated HR Modules",
        data_blocks=data_blocks,
        file_prefix="Consolidated_HR_Leadership_Report"
    )