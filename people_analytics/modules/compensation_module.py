# ============================================
# modules/compensation_module.py — v2.0 | Smart Insights + Company vs Market + Executive PDF
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.pdf_auto_exporter import export_module_report


def run_compensation_module():
    # =========================
    # 💰 Header
    # =========================
    st.markdown("""
    <div style="padding:20px; border-radius:12px;
                background:linear-gradient(90deg,#14532D,#22C55E);
                color:white; text-align:center; margin-bottom:20px;">
        <h2 style="margin:0;">💰 Compensation Analytics</h2>
        <p style="font-size:14px; margin-top:6px;">
            Analyze pay structure, bonus distribution, gender gap, and benchmark your organization
            against market medians.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📄 Step 1 — Download Templates
    # =========================
    st.subheader("📄 Step 1 — Download Templates")

    emp_template = pd.DataFrame([{
        "EmployeeID": "E1001", "Gender": "Male", "Department": "Finance",
        "JobRole": "Analyst", "JobLevel": "Analyst",
        "CTC": 600000, "Bonus": 50000, "PerformanceRating": 3
    }])
    bench_template = pd.DataFrame([{
        "JobRole": "Analyst", "JobLevel": "Analyst",
        "MarketMedianCTC": 650000
    }])

    c1, c2 = st.columns(2)
    with c1:
        render_download_template("Internal Data Template", emp_template, "Internal_Template.csv")
    with c2:
        render_download_template("Benchmark Data Template", bench_template, "Benchmark_Template.csv")

    # =========================
    # 📤 Step 2 — Upload Data
    # =========================
    st.subheader("📤 Step 2 — Upload Data")
    c1, c2 = st.columns(2)
    emp_file = c1.file_uploader(
        "Upload Internal Data", type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
    )
    bench_file = c2.file_uploader(
        "Upload Benchmark Data (optional)", type=["csv", "xlsx", "text", "plain", "application/vnd.ms-excel"]
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

    if bench_file:
        try:
            if bench_file.name.endswith(".csv"):
                bench_df = pd.read_csv(bench_file)
            else:
                bench_df = pd.read_excel(bench_file, engine="openpyxl")
            st.success("✅ Benchmark file uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading benchmark file: {e}")
            bench_df = None
    else:
        bench_df = None

    # =========================
    # 🧮 Step 3 — Validation
    # =========================
    required = ["EmployeeID", "Gender", "Department", "JobRole", "JobLevel", "CTC", "Bonus"]
    missing = [c for c in required if c not in emp_df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.dataframe(emp_df.head(), use_container_width=True)

    # =========================
    # 📊 Step 4 — Core Metrics
    # =========================
    st.subheader("📊 Compensation Metrics")

    emp_df["BonusPct"] = (emp_df["Bonus"] / emp_df["CTC"]) * 100
    avg_ctc = emp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index()
    avg_bonus = emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index()
    gender_gap = emp_df.groupby("Gender", observed=True)["CTC"].mean().reset_index()

    # --- Visuals ---
    c1, c2 = st.columns(2)
    with c1:
        fig_ctc = px.bar(avg_ctc, x="JobLevel", y="CTC", text="CTC", color="JobLevel",
                         title="Average CTC by Job Level")
        fig_ctc.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        st.plotly_chart(fig_ctc, use_container_width=True)
    with c2:
        fig_bonus = px.bar(avg_bonus, x="JobLevel", y="BonusPct", text="BonusPct", color="JobLevel",
                           title="Average Bonus % by Job Level")
        fig_bonus.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_bonus, use_container_width=True)

    st.subheader("⚖️ Gender Pay Gap")
    fig_gender = px.bar(gender_gap, x="Gender", y="CTC", text="CTC", color="Gender",
                        title="Average CTC by Gender")
    fig_gender.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    st.plotly_chart(fig_gender, use_container_width=True)

    gap = abs(gender_gap["CTC"].max() - gender_gap["CTC"].min())
    st.info(f"💡 Gender Pay Gap: ₹{gap:,.0f}")

    # =========================
    # 🧭 Step 5 — Company vs Market Comparison
    # =========================
    if bench_df is not None and not bench_df.empty:
        merged = emp_df.merge(bench_df, on=["JobRole", "JobLevel"], how="left")
        merged["MarketMedianCTC"] = pd.to_numeric(merged["MarketMedianCTC"], errors="coerce").fillna(0)
        merged["DiffPct"] = ((merged["CTC"] - merged["MarketMedianCTC"]) /
                             merged["MarketMedianCTC"].replace(0, pd.NA)) * 100
        merged["DiffPct"] = merged["DiffPct"].fillna(0)

        # Summary
        comp_summary = merged.groupby("JobLevel", observed=True)[["CTC", "MarketMedianCTC", "DiffPct"]].mean().reset_index()

        st.subheader("🏦 Company vs Market Median (CTC Comparison)")
        st.dataframe(comp_summary.round(2), use_container_width=True)

        # ✅ Fixed comparison chart — shows both bars clearly
        fig_comp = px.bar(
            comp_summary.melt(id_vars="JobLevel", value_vars=["CTC", "MarketMedianCTC"],
                              var_name="Source", value_name="CTC_Value"),
            x="JobLevel", y="CTC_Value", color="Source", barmode="group",
            text_auto=".2s", title="Company vs Market Median (CTC)"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Insights
        best_level = comp_summary.loc[comp_summary["DiffPct"].idxmax(), "JobLevel"]
        worst_level = comp_summary.loc[comp_summary["DiffPct"].idxmin(), "JobLevel"]
        st.markdown(f"""
        <div style="background:#0F172A;padding:10px 15px;border-radius:8px;margin-top:10px;">
        <b>💡 Insights:</b><br>
        • Highest market premium: <b>{best_level}</b><br>
        • Most underpaid level vs market: <b>{worst_level}</b><br>
        • Avg company-to-market gap: <b>{comp_summary['DiffPct'].mean():.1f}%</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Benchmark data not uploaded or empty. Skipping market comparison.")
        comp_summary = pd.DataFrame()

    # =========================
    # 📄 Step 6 — Export Executive Report
    # =========================
    st.markdown("---")
    st.subheader("📄 Step 6 — Export Executive Report")

    data_blocks = [
        {
            "title": "Compensation Metrics Overview",
            "desc": "Average CTC, Bonus %, and Gender Gap Summary.",
            "df": avg_ctc,
            "insights": [
                "Bonus % and pay distribution visualized across levels.",
                f"Gender pay gap observed at ₹{gap:,.0f}."
            ]
        },
        {
            "title": "Company vs Market Comparison",
            "desc": "Comparison of internal compensation to external benchmarks.",
            "df": comp_summary,
            "insights": [
                "Market positioning assessed by job level.",
                "Identified high and low parity zones."
            ]
        }
    ]

    export_module_report(
        report_title="Compensation Analytics Executive Report",
        module_name="Compensation",
        data_blocks=data_blocks,
        filename_prefix="Compensation"
    )