# modules/engagement_module.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template

# ==============================
# Engagement Analytics Module (v1.0)
# ==============================

def run_engagement_module():
    st.header("💬 Employee Engagement Analytics")

    st.markdown("""
    This module analyzes employee engagement survey data — from overall engagement levels 
    to departmental heatmaps — based on a standard 1–5 Likert scale (1 = Strongly Disagree, 5 = Strongly Agree).
    """)

    # =========================
    # 📥 Download Template
    # =========================
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

    # =========================
    # 📤 Upload Completed Survey
    # =========================
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

    # =========================
    # 🧮 Validate and Analyze
    # =========================
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender"]
    question_cols = [col for col in df.columns if col.startswith("Q")]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return
    if not question_cols:
        st.error("No question columns (Q1, Q2, etc.) found.")
        return

    # Convert to numeric
    for q in question_cols:
        df[q] = pd.to_numeric(df[q], errors="coerce").clip(1, 5)

    # Drop invalid rows
    df = df.dropna(subset=question_cols, how="all")
    if df.empty:
        st.error("No valid responses found after cleaning.")
        return

    # =========================
    # 📊 Engagement Index Calculations
    # =========================
    df["EngagementIndex"] = df[question_cols].mean(axis=1)
    df["EngagementCategory"] = pd.cut(
        df["EngagementIndex"],
        bins=[0, 2.5, 3.5, 5],
        labels=["Low", "Moderate", "High"],
        include_lowest=True,
    )

    st.metric("Overall Engagement Index (1–5)", f"{df['EngagementIndex'].mean():.2f}")

    # Department summary
    st.subheader("🏢 Engagement by Department")
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
    st.plotly_chart(bar, use_container_width=True)

    # Category distribution
    st.subheader("🎯 Engagement Level Distribution")
    cat_counts = df["EngagementCategory"].value_counts().reindex(["High", "Moderate", "Low"]).fillna(0).astype(int)
    pie = px.pie(values=cat_counts.values, names=cat_counts.index, title="Engagement Category Breakdown")
    st.plotly_chart(pie, use_container_width=True)

    # =========================
    # 📤 Export Processed File
    # =========================
    st.subheader("📤 Step 3 — Export Processed Survey")
    export_df = df[["EmployeeID", "Department", "JobLevel", "Gender", *question_cols, "EngagementIndex", "EngagementCategory"]]
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Processed Engagement Data (CSV)",
        data=csv_bytes,
        file_name="Engagement_Processed.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success("✅ Engagement analysis complete!")