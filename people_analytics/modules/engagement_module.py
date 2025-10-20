# ============================================
# modules/engagement_module.py — v1.2 | PDF Export + Insights
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_helper import render_pdf_download_button

def run_engagement_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#1E3A8A,#3B82F6);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💬 Employee Engagement Analytics</h2>
        <p style="font-size:14px;margin-top:6px;">
        Analyze engagement survey data, understand team morale, and identify focus areas for HR interventions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Template Download ---
    st.subheader("📄 Step 1 — Download Survey Template")
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

    # --- Upload ---
    st.subheader("📤 Step 2 — Upload Completed Survey")
    uploaded = st.file_uploader("Upload filled survey (CSV only)", type=["csv"])
    if not uploaded:
        st.info("Please upload a survey file to continue.")
        return

    try:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender"]
    question_cols = [col for col in df.columns if col.startswith("Q")]
    if not question_cols:
        st.error("No question columns (Q1, Q2...) found.")
        return

    for q in question_cols:
        df[q] = pd.to_numeric(df[q], errors="coerce").clip(1, 5)
    df = df.dropna(subset=question_cols, how="all")

    df["EngagementIndex"] = df[question_cols].mean(axis=1)
    df["EngagementCategory"] = pd.cut(df["EngagementIndex"],
                                      bins=[0, 2.5, 3.5, 5],
                                      labels=["Low", "Moderate", "High"],
                                      include_lowest=True)

    st.metric("🌟 Overall Engagement Index", f"{df['EngagementIndex'].mean():.2f}")

    with st.expander("🏢 A. Engagement by Department", expanded=True):
        dept_summary = df.groupby("Department", observed=True)["EngagementIndex"].mean().round(2).reset_index()
        bar = px.bar(dept_summary, x="Department", y="EngagementIndex", color="Department", text="EngagementIndex")
        bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(bar, use_container_width=True)

        top_dept = dept_summary.loc[dept_summary["EngagementIndex"].idxmax(), "Department"]
        low_dept = dept_summary.loc[dept_summary["EngagementIndex"].idxmin(), "Department"]
        diff = dept_summary["EngagementIndex"].max() - dept_summary["EngagementIndex"].min()

        st.markdown(f"""
        <div style="background:#0F172A;padding:10px 15px;border-radius:8px;margin-top:10px;">
        <b>🧠 Insights:</b><br>
        • Most engaged: <b>{top_dept}</b><br>
        • Lowest engagement: <b>{low_dept}</b><br>
        • Department gap: <b>{diff:.2f}</b> points
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🎯 B. Engagement Level Distribution", expanded=True):
        cat_counts = df["EngagementCategory"].value_counts().reindex(["High", "Moderate", "Low"]).fillna(0).astype(int)
        pie = px.pie(values=cat_counts.values, names=cat_counts.index, title="Engagement Level Breakdown")
        st.plotly_chart(pie, use_container_width=True)

        st.markdown(f"""
        <div style="background:#0F172A;padding:10px 15px;border-radius:8px;margin-top:10px;">
        <b>💡 Insights:</b><br>
        • {cat_counts.get('High', 0)} employees are highly engaged.<br>
        • {cat_counts.get('Moderate', 0)} are moderately engaged.<br>
        • {cat_counts.get('Low', 0)} show low engagement levels.
        </div>
        """, unsafe_allow_html=True)

    # --- PDF Export ---
    st.subheader("📄 Step 3 — Export Engagement Report")
    html_summary = f"""
    <h2>Engagement Analytics Summary</h2>
    <p>This report summarizes engagement levels across departments and highlights priority zones for HR focus.</p>
    <div class='summary'>
    <p><b>Most Engaged Department:</b> {top_dept}<br>
    <b>Lowest Engaged Department:</b> {low_dept}<br>
    <b>Score Gap:</b> {diff:.2f} points</p>
    </div>
    """
    render_pdf_download_button("Engagement Analytics Report", html_summary, "Engagement_Report")

    st.markdown("""
    <hr style="border:1px solid #1E3A8A;margin-top:40px;"/>
    <div style="text-align:center;color:#9CA3AF;font-size:13px;">
        Prepared with ❤️ by <b>Amlan Mishra</b> | © 2025 HR Tech Portfolio
    </div>
    """, unsafe_allow_html=True)