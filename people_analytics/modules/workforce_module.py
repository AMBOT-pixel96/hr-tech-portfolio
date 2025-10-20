# ============================================
# modules/workforce_module.py — v1.0
# Workforce & Talent Analytics Module
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.template_helper import render_download_template
from utils.pdf_helper import render_pdf_download_button

def run_workforce_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#0F766E,#14B8A6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">🏢 Workforce & Talent Analytics</h2>
        <p style="font-size:14px;margin-top:6px;">
        Understand workforce structure, span of control, headcount mix, and cost distribution — all from one consolidated view.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================
    # Step 1: Download Template
    # ==========================
    st.subheader("📄 Step 1 — Download Workforce Data Template")

    sample_data = pd.DataFrame({
        "EmployeeID": ["E1001", "E1002", "E1003"],
        "Department": ["Finance", "IT", "HR"],
        "JobLevel": ["Analyst", "Manager", "Director"],
        "Gender": ["Male", "Female", "Male"],
        "ManagerID": ["M001", "M002", "M003"],
        "CTC": [600000, 1200000, 2500000],
        "Skills": ["Excel;PowerBI", "Python;SQL", "Leadership;HRIS"]
    })
    render_download_template("Workforce Data Template", sample_data, "Workforce_Template.csv")

    # ==========================
    # Step 2: Upload Data
    # ==========================
    st.subheader("📤 Step 2 — Upload Workforce Dataset")
    uploaded = st.file_uploader("Upload completed workforce dataset (CSV)", type=["csv"])

    if not uploaded:
        st.info("Please upload your workforce dataset to continue.")
        return

    try:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # ==========================
    # Step 3: Validation
    # ==========================
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender", "ManagerID", "CTC"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    df["CTC"] = pd.to_numeric(df["CTC"], errors="coerce")
    df = df.dropna(subset=["CTC"])

    # ==========================
    # Step 4: Headcount Overview
    # ==========================
    st.subheader("👥 Workforce Composition Overview")

    total_employees = len(df)
    avg_ctc = df["CTC"].mean() / 1e5
    gender_split = df["Gender"].value_counts(normalize=True).mul(100).round(1).to_dict()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Headcount", f"{total_employees:,}")
    col2.metric("Average CTC (₹ Lakhs)", f"{avg_ctc:.2f}")
    col3.metric("Gender Split", f"♂ {gender_split.get('Male',0)}% | ♀ {gender_split.get('Female',0)}%")

    # --- Departmental Headcount
    st.markdown("#### 🏢 Headcount by Department")
    dept_count = df["Department"].value_counts().reset_index()
    dept_count.columns = ["Department", "Headcount"]
    fig_dept = px.bar(dept_count, x="Department", y="Headcount", color="Department",
                      text="Headcount", title="Headcount by Department")
    fig_dept.update_traces(textposition="outside")
    st.plotly_chart(fig_dept, use_container_width=True)

    # --- Job Level Pyramid
    st.markdown("#### 🪜 Organization Pyramid by Job Level")
    level_count = df["JobLevel"].value_counts().reset_index()
    level_count.columns = ["JobLevel", "Headcount"]
    level_count = level_count.sort_values(by="Headcount", ascending=True)

    fig_pyramid = go.Figure(go.Bar(
        y=level_count["JobLevel"],
        x=level_count["Headcount"],
        orientation="h",
        text=level_count["Headcount"],
        textposition="outside",
        marker=dict(color="#0EA5E9")
    ))
    fig_pyramid.update_layout(title="Organization Pyramid (Headcount by Job Level)",
                              xaxis_title="Headcount", yaxis_title="")
    st.plotly_chart(fig_pyramid, use_container_width=True)

    # ==========================
    # Step 5: Span of Control
    # ==========================
    st.subheader("🧩 Span of Control (Managers vs Directs)")
    span = df.groupby("ManagerID", observed=True)["EmployeeID"].count().reset_index()
    span.columns = ["ManagerID", "DirectReports"]

    avg_span = span["DirectReports"].mean().round(1)
    st.metric("Average Span of Control", f"{avg_span} reports per manager")

    span_fig = px.histogram(span, x="DirectReports", nbins=10, color_discrete_sequence=["#14B8A6"],
                            title="Span of Control Distribution")
    st.plotly_chart(span_fig, use_container_width=True)

    # ==========================
    # Step 6: Workforce Cost Analysis
    # ==========================
    st.subheader("💰 Workforce Cost Analysis")

    dept_cost = df.groupby("Department", observed=True)["CTC"].sum().reset_index()
    dept_cost["CTC_Lakhs"] = (dept_cost["CTC"] / 1e5).round(2)
    fig_cost = px.pie(dept_cost, values="CTC_Lakhs", names="Department", title="Departmental Cost Share (₹ Lakhs)")
    st.plotly_chart(fig_cost, use_container_width=True)

    # ==========================
    # Step 7: Skills Inventory (Text-based Summary)
    # ==========================
    st.subheader("🧠 Skills Inventory Summary")
    if "Skills" in df.columns:
        skill_series = df["Skills"].dropna().str.split(";").explode().str.strip()
        skill_count = skill_series.value_counts().head(10)
        st.bar_chart(skill_count)
        st.caption("Top 10 most common skills in the organization.")
    else:
        st.info("No 'Skills' column found. Skipping skills inventory section.")

    # ==========================
    # Step 8: PDF Export
    # ==========================
    st.subheader("📄 Step 3 — Export Workforce Summary Report")

    html_summary = f"""
    <h2>Workforce Summary Report</h2>
    <p><b>Total Headcount:</b> {total_employees:,}<br>
    <b>Average CTC:</b> ₹{avg_ctc:.2f} LPA<br>
    <b>Average Span:</b> {avg_span} direct reports<br>
    <b>Gender Mix:</b> {gender_split.get('Male',0)}% Male / {gender_split.get('Female',0)}% Female</p>
    """
    render_pdf_download_button("Workforce Summary Report", html_summary, "Workforce_Report")

    st.markdown("""
    <hr style="border:1px solid #0F766E;margin-top:40px;"/>
    <div style="text-align:center;color:#9CA3AF;font-size:13px;">
        Prepared with ❤️ by <b>Amlan Mishra</b> | © 2025 HR Tech Portfolio
    </div>
    """, unsafe_allow_html=True)