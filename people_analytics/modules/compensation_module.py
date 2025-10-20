# ============================================
# modules/compensation_module.py — v1.3 | CSV Fix + Smart PDF Export + Insights
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report

def run_compensation_module():
    st.markdown("""
    <div style="padding:20px; border-radius:12px; background:linear-gradient(90deg,#14532D,#22C55E);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💰 Compensation Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">Analyze pay structure, bonus distribution, gender gap, and market benchmarking — all from your HRIS data.</p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📄 Step 1 — Download Templates
    # =========================
    st.subheader("📄 Step 1 — Download Templates")

    emp_template = pd.DataFrame([{
        "EmployeeID": "E1001",
        "Gender": "Male",
        "Department": "Finance",
        "JobRole": "Analyst",
        "JobLevel": "Analyst",
        "CTC": 600000,
        "Bonus": 50000,
        "PerformanceRating": 3
    }])

    bench_template = pd.DataFrame([{
        "JobRole": "Analyst",
        "JobLevel": "Analyst",
        "MarketMedianCTC": 650000
    }])

    col1, col2 = st.columns(2)
    with col1:
        render_download_template("Internal Data Template", emp_template, "Internal_Template.csv")
    with col2:
        render_download_template("Benchmark Data Template", bench_template, "Benchmark_Template.csv")

    # =========================
    # 📤 Step 2 — Upload Data
    # =========================
    st.subheader("📤 Step 2 — Upload Compensation Data")
    col1, col2 = st.columns(2)

    emp_file = col1.file_uploader(
        "Upload Internal Data",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )
    bench_file = col2.file_uploader(
        "Upload Benchmark Data (optional)",
        type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )

    if emp_file is None:
        st.info("Please upload your internal compensation file to begin analysis.")
        return

    try:
        if emp_file.name.endswith(".csv"):
            emp_df = pd.read_csv(emp_file)
        else:
            emp_df = pd.read_excel(emp_file, engine="openpyxl")
        st.success("✅ Internal file uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading internal file: {e}")
        return

    bench_df = None
    if bench_file:
        try:
            if bench_file.name.endswith(".csv"):
                bench_df = pd.read_csv(bench_file)
            else:
                bench_df = pd.read_excel(bench_file, engine="openpyxl")
            st.success("✅ Benchmark file uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading benchmark file: {e}")

    # =========================
    # 🧮 Step 3 — Validation
    # =========================
    required_cols = ["EmployeeID", "Gender", "Department", "JobRole", "JobLevel", "CTC", "Bonus", "PerformanceRating"]
    missing = [col for col in required_cols if col not in emp_df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.dataframe(emp_df.head(), use_container_width=True)

    # =========================
    # 📊 Step 4 — Core Metrics
    # =========================
    st.subheader("📈 Compensation Metrics")

    emp_df["BonusPct"] = (emp_df["Bonus"] / emp_df["CTC"]) * 100
    avg_ctc = emp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index()
    avg_bonus = emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index()
    gender_gap = emp_df.groupby("Gender", observed=True)["CTC"].mean().reset_index()

    # --- Visuals ---
    col1, col2 = st.columns(2)
    with col1:
        fig_ctc = px.bar(avg_ctc, x="JobLevel", y="CTC", text="CTC", color="JobLevel", title="Average CTC by Job Level")
        fig_ctc.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        st.plotly_chart(fig_ctc, use_container_width=True)

    with col2:
        fig_bonus = px.bar(avg_bonus, x="JobLevel", y="BonusPct", text="BonusPct", color="JobLevel", title="Average Bonus % by Job Level")
        fig_bonus.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_bonus, use_container_width=True)

    st.subheader("⚖️ Gender Pay Gap")
    fig_gender = px.bar(gender_gap, x="Gender", y="CTC", text="CTC", color="Gender", title="Average CTC by Gender")
    fig_gender.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    st.plotly_chart(fig_gender, use_container_width=True)

    gap = abs(gender_gap["CTC"].max() - gender_gap["CTC"].min())
    st.info(f"💡 Gender Pay Gap: ₹{gap:,.0f}")

    # =========================
    # 🧭 Step 5 — Market Comparison
    # =========================
    if bench_df is not None:
        merged = emp_df.merge(bench_df, on=["JobRole", "JobLevel"], how="left")
        merged["MarketMedianCTC"] = pd.to_numeric(merged["MarketMedianCTC"], errors="coerce").fillna(0)
        merged["DiffPct"] = ((merged["CTC"] - merged["MarketMedianCTC"]) / merged["MarketMedianCTC"].replace(0, pd.NA)) * 100
        merged["DiffPct"] = merged["DiffPct"].fillna(0)

        comp_summary = merged.groupby("JobLevel", observed=True)["DiffPct"].mean().round(2).reset_index()
        comp_summary = comp_summary.sort_values(by="DiffPct", ascending=False)

        st.subheader("🏦 Company vs Market Median (CTC % Difference)")
        st.dataframe(comp_summary, use_container_width=True)
    else:
        st.warning("⚠️ Benchmark data not uploaded. Skipping market comparison.")
        comp_summary = pd.DataFrame()

    # =========================
    # 📄 Step 6 — Executive PDF Export
    # =========================
    data_blocks = [
        {
            "title": "Compensation Metrics",
            "desc": "Average pay levels and bonus percentages across job levels.",
            "df": avg_ctc,
            "insights": [
                f"Average CTC range: ₹{avg_ctc['CTC'].min():,.0f} – ₹{avg_ctc['CTC'].max():,.0f}",
                f"Highest bonus level: {avg_bonus.loc[avg_bonus['BonusPct'].idxmax(), 'JobLevel']} ({avg_bonus['BonusPct'].max():.1f}%)"
            ]
        },
        {
            "title": "Gender Pay Gap",
            "desc": "Comparison of average CTC by gender.",
            "df": gender_gap,
            "insights": [f"Gender pay gap: ₹{gap:,.0f}"]
        },
        {
            "title": "Market Benchmarking",
            "desc": "Internal vs external CTC differences.",
            "df": comp_summary,
            "insights": [
                "Market comparison performed where benchmark data available.",
                f"Highest variance job level: {comp_summary.iloc[0]['JobLevel'] if not comp_summary.empty else 'N/A'}"
            ]
        }
    ]

    export_module_report(
        report_title="Compensation Analytics Executive Report",
        module_name="Compensation",
        data_blocks=data_blocks,
        filename_prefix="Compensation"
    )