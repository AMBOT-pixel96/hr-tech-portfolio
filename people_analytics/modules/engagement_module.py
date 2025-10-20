# ============================================
# modules/engagement_module.py — v1.1 Polished + Insight Summaries
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template

# ==============================
# Engagement Analytics Module
# ==============================

def run_engagement_module():
    # --- Header Banner ---
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💬 Employee Engagement Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
        Analyze engagement trends from survey data — spot high-energy teams, understand 
        disengagement roots, and guide people-first actions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Step 1: Download Template ---
    st.subheader("📄 Step 1 — Download Engagement Survey Template")

    sample_data = pd.DataFrame({
        "EmployeeID": ["E1001", "E1002"],
        "Department": ["Finance", "HR"],
        "JobLevel": ["Analyst", "Manager"],
        "Gender": ["Male", "Female"],
        "Q1": [5, 4],
        "Q2": [4, 3],
        "Q3": [3, 4],
        "Q4": [5, 5],
        "Q5": [4, 4],
    })
    render_download_template("Engagement Survey Template", sample_data, "Engagement_Survey_Template.csv")

    # --- Step 2: Upload Completed Survey ---
    st.subheader("📤 Step 2 — Upload Completed Survey File")
    uploaded = st.file_uploader("Upload filled survey (CSV only)", type=["csv"])

    if not uploaded:
        st.info("Please upload the completed survey file to proceed.")
        return

    try:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # --- Validation ---
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender"]
    question_cols = [col for col in df.columns if col.startswith("Q")]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return
    if not question_cols:
        st.error("No question columns (Q1, Q2, etc.) found.")
        return

    # Convert to numeric safely
    for q in question_cols:
        df[q] = pd.to_numeric(df[q], errors="coerce").clip(1, 5)

    # Drop invalid rows
    df = df.dropna(subset=question_cols, how="all")
    if df.empty:
        st.error("No valid responses found after cleaning.")
        return

    # --- Step 3: Engagement Index Calculation ---
    df["EngagementIndex"] = df[question_cols].mean(axis=1)
    df["EngagementCategory"] = pd.cut(
        df["EngagementIndex"],
        bins=[0, 2.5, 3.5, 5],
        labels=["Low", "Moderate", "High"],
        include_lowest=True,
    )

    # --- Summary Metrics ---
    st.metric("🌟 Overall Engagement Index (1–5)", f"{df['EngagementIndex'].mean():.2f}")

    with st.expander("🏢 A. Engagement by Department", expanded=True):
        dept_summary = df.groupby("Department", observed=True)["EngagementIndex"].mean().round(2).reset_index()
        st.dataframe(dept_summary, use_container_width=True)

        bar = px.bar(
            dept_summary,
            x="Department",
            y="EngagementIndex",
            text="EngagementIndex",
            title="Average Engagement by Department",
            color="Department",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(bar, use_container_width=True)

        # Insight Summary
        top_dept = dept_summary.loc[dept_summary["EngagementIndex"].idxmax(), "Department"]
        low_dept = dept_summary.loc[dept_summary["EngagementIndex"].idxmin(), "Department"]
        diff = dept_summary["EngagementIndex"].max() - dept_summary["EngagementIndex"].min()
        st.markdown(f"""
        <div style="background:#0F172A; padding:10px 15px; border-radius:8px; margin-top:10px;">
        <b>🧠 Insight Summary:</b><br>
        • Most engaged department: <b>{top_dept}</b><br>
        • Lowest engagement seen in: <b>{low_dept}</b><br>
        • Score gap between top and bottom: <b>{diff:.2f}</b> points
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🎯 B. Engagement Level Distribution", expanded=True):
        cat_counts = df["EngagementCategory"].value_counts().reindex(["High", "Moderate", "Low"]).fillna(0).astype(int)
        pie = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            title="Engagement Category Breakdown",
            color=cat_counts.index,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(pie, use_container_width=True)

        st.markdown(f"""
        <div style="background:#0F172A; padding:10px 15px; border-radius:8px; margin-top:10px;">
        <b>💡 Insight Summary:</b><br>
        • {cat_counts.get('High', 0)} employees show strong engagement.<br>
        • {cat_counts.get('Moderate', 0)} are moderately engaged.<br>
        • {cat_counts.get('Low', 0)} indicate low engagement — potential concern area.
        </div>
        """, unsafe_allow_html=True)

    # --- Step 4: Export Processed File ---
    with st.expander("📤 C. Export Processed Survey Results", expanded=False):
        export_df = df[
            ["EmployeeID", "Department", "JobLevel", "Gender", *question_cols, "EngagementIndex", "EngagementCategory"]
        ]
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Processed Engagement Data (CSV)",
            data=csv_bytes,
            file_name="Engagement_Processed.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.success("✅ Engagement analysis complete!")

    # --- Footer ---
    st.markdown("""
    <hr style="border:1px solid #1E3A8A; margin-top:40px;"/>
    <div style="text-align:center; color:#9CA3AF; font-size:13px;">
        Prepared with ❤️ by <b>Amlan Mishra</b> | © 2025 HR Tech Portfolio
    </div>
    """, unsafe_allow_html=True)