# ============================================
# modules/engagement_module.py — v1.1 | CSV Upload Fix + Polished Layout
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template

def run_engagement_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#1E40AF,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💬 Employee Engagement Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">Analyze engagement survey responses — discover sentiment patterns, departmental scores, and organizational engagement trends.</p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📄 Step 1 — Download Template
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
    # 📤 Step 2 — Upload Completed Survey
    # =========================
    st.subheader("📤 Step 2 — Upload Completed Survey File")
    uploaded = st.file_uploader(
        "Upload filled survey file (CSV, Excel, or Text)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )

    if not uploaded:
        st.info("Please upload the completed survey file to proceed.")
        return

    # --- Read uploaded data ---
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, engine="openpyxl")
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # =========================
    # 🧮 Step 3 — Validate & Analyze
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

    # --- Numeric Conversion ---
    for q in question_cols:
        df[q] = pd.to_numeric(df[q], errors="coerce").clip(1, 5)

    # Drop invalid rows
    df = df.dropna(subset=question_cols, how="all")
    if df.empty:
        st.error("No valid responses found after cleaning.")
        return

    # --- Engagement Index ---
    df["EngagementIndex"] = df[question_cols].mean(axis=1)
    df["EngagementCategory"] = pd.cut(
        df["EngagementIndex"],
        bins=[0, 2.5, 3.5, 5],
        labels=["Low", "Moderate", "High"],
        include_lowest=True,
    )

    st.metric("Overall Engagement Index (1–5)", f"{df['EngagementIndex'].mean():.2f}")

    # =========================
    # 📊 Step 4 — Visualizations
    # =========================

    # Department Summary
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

    # Category Breakdown
    st.subheader("🎯 Engagement Level Distribution")
    cat_counts = df["EngagementCategory"].value_counts().reindex(["High", "Moderate", "Low"]).fillna(0).astype(int)
    pie = px.pie(values=cat_counts.values, names=cat_counts.index, title="Engagement Category Breakdown")
    st.plotly_chart(pie, use_container_width=True)

    # =========================
    # 📤 Step 5 — Export Processed File
    # =========================
    st.subheader("📤 Step 5 — Export Processed Survey")
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

    st.markdown("""
    <hr style="border:1px solid #1E40AF;margin-top:40px;"/>
    <div style="text-align:center;color:#9CA3AF;font-size:13px;">
        Prepared with ❤️ by <b>Amlan Mishra</b> | © 2025 HR Tech Portfolio
    </div>
    """, unsafe_allow_html=True)