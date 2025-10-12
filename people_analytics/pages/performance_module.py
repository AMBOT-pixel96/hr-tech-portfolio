import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==============================
# Performance Analytics Module
# ==============================

def run_performance_module():
    st.header("🏆 Performance Analytics")

    st.markdown("""
    Analyze employee performance patterns, identify high-potential clusters, 
    and understand how ratings relate to pay and skills.
    """)

    # --- Upload Data ---
    st.subheader("📤 Upload Performance Data")
    perf_file = st.file_uploader("Upload Performance Data (CSV or Excel)", type=["csv", "xlsx"])

    if perf_file is None:
        st.info("Please upload a dataset to continue.")
        return

    # --- Read File ---
    if perf_file.name.endswith(".csv"):
        df = pd.read_csv(perf_file)
    else:
        df = pd.read_excel(perf_file, engine="openpyxl")

    st.success("✅ File uploaded successfully!")

    # --- Basic Validation ---
    required_cols = ["EmployeeID", "Department", "JobLevel", "Gender", "PerformanceRating", "CTC"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.dataframe(df.head(), use_container_width=True)

    # ==================================
    # 📊 METRICS A — Descriptive Insights
    # ==================================
    st.subheader("📈 A. Performance Distribution Insights")

    # 1. Bell Curve
    bell_fig = px.histogram(
        df,
        x="PerformanceRating",
        nbins=5,
        color_discrete_sequence=["#4B9CD3"],
        title="Performance Rating Bell Curve"
    )
    st.plotly_chart(bell_fig, use_container_width=True)

    # 2. Rating by Department
    dept_fig = px.box(
        df,
        x="Department",
        y="PerformanceRating",
        color="Department",
        title="Performance Ratings by Department"
    )
    st.plotly_chart(dept_fig, use_container_width=True)

    # 3. Rating by Job Level
    level_fig = px.box(
        df,
        x="JobLevel",
        y="PerformanceRating",
        color="JobLevel",
        title="Performance Ratings by Job Level"
    )
    st.plotly_chart(level_fig, use_container_width=True)

    # 4. Rating by Gender
    gender_avg = df.groupby("Gender")["PerformanceRating"].mean().reset_index()
    gender_fig = px.bar(
        gender_avg,
        x="Gender",
        y="PerformanceRating",
        color="Gender",
        text="PerformanceRating",
        title="Average Performance Rating by Gender"
    )
    gender_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(gender_fig, use_container_width=True)

    # ==================================
    # ⚙️ METRICS B — Performance vs Skills (Future)
    # ==================================
    st.subheader("🧠 B. Performance vs Skills")
    st.info("This section will integrate skill matrix data in v2 (Talent Insights module).")

    # ==================================
    # 💰 METRICS C — Performance vs Pay
    # ==================================
    st.subheader("💰 C. Performance vs Pay")

    pay_fig = px.box(
        df,
        x="PerformanceRating",
        y="CTC",
        color="PerformanceRating",
        title="CTC by Performance Rating"
    )
    st.plotly_chart(pay_fig, use_container_width=True)

    perf_pay_avg = df.groupby("PerformanceRating")["CTC"].mean().reset_index()
    perf_pay_avg["CTC (₹ Lakhs)"] = (perf_pay_avg["CTC"] / 1e5).round(2)

    st.dataframe(perf_pay_avg[["PerformanceRating", "CTC (₹ Lakhs)"]], use_container_width=True)

    st.success("✅ Performance analytics generated successfully!")

# --- End of module ---
