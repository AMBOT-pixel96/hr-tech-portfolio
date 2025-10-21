# ============================================
# modules/compensation_module.py — v1.3 | Smart PDF Export
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
        <p style="font-size:14px; margin-top:6px;">
            Analyze pay structure, bonus distribution, gender gap, and market benchmarking.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Step 1: Template ---
    st.subheader("📄 Step 1 — Download Templates")
    emp_template = pd.DataFrame([{"EmployeeID":"E1001","Gender":"Male","Department":"Finance","JobRole":"Analyst","JobLevel":"Analyst","CTC":600000,"Bonus":50000,"PerformanceRating":3}])
    bench_template = pd.DataFrame([{"JobRole":"Analyst","JobLevel":"Analyst","MarketMedianCTC":650000}])
    c1,c2 = st.columns(2)
    with c1: render_download_template("Internal Data Template", emp_template, "Internal_Template.csv")
    with c2: render_download_template("Benchmark Data Template", bench_template, "Benchmark_Template.csv")

    # --- Step 2: Upload ---
    st.subheader("📤 Step 2 — Upload Data")
    c1,c2 = st.columns(2)
    emp_file = c1.file_uploader("Upload Internal Data", type=["csv","xlsx"])
    bench_file = c2.file_uploader("Upload Benchmark Data (optional)", type=["csv","xlsx"])

    if emp_file is None:
        st.info("Please upload your internal compensation file to begin analysis.")
        return
    emp_df = pd.read_csv(emp_file) if emp_file.name.endswith(".csv") else pd.read_excel(emp_file, engine="openpyxl")
    bench_df = pd.read_csv(bench_file) if bench_file and bench_file.name.endswith(".csv") else (pd.read_excel(bench_file, engine="openpyxl") if bench_file else None)
    st.success("✅ Data uploaded successfully!")

    # --- Step 3: Core Metrics ---
    emp_df["BonusPct"] = (emp_df["Bonus"]/emp_df["CTC"])*100
    avg_ctc = emp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index()
    avg_bonus = emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index()
    gender_gap = emp_df.groupby("Gender", observed=True)["CTC"].mean().reset_index()

    st.subheader("📈 Compensation Metrics")
    st.plotly_chart(px.bar(avg_ctc, x="JobLevel", y="CTC", text="CTC", color="JobLevel"), use_container_width=True)
    st.plotly_chart(px.bar(avg_bonus, x="JobLevel", y="BonusPct", text="BonusPct", color="JobLevel"), use_container_width=True)
    st.plotly_chart(px.bar(gender_gap, x="Gender", y="CTC", text="CTC", color="Gender"), use_container_width=True)

    # --- Step 4: Market Comparison ---
    if bench_df is not None:
        merged = emp_df.merge(bench_df, on=["JobRole","JobLevel"], how="left")
        merged["DiffPct"] = ((merged["CTC"]-merged["MarketMedianCTC"])/merged["MarketMedianCTC"].replace(0,pd.NA))*100
        comp_summary = merged.groupby("JobLevel", observed=True)["DiffPct"].mean().round(2).reset_index()
        fig_comp = px.bar(comp_summary, x="JobLevel", y="DiffPct", text="DiffPct", color="JobLevel",
                          title="Company vs Market Median — % Difference")
        fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- Step 5: Export ---
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    data_blocks = [{
        "title": "Compensation Overview",
        "desc": "Pay, bonus, gender gap, and market comparison summarized.",
        "df": avg_ctc,
        "insights": ["Average pay and bonus distribution by level.","Gender pay gap and market variance summarized."]
    }]
    export_module_report("Compensation Analytics Executive Report","Compensation",data_blocks,"Compensation")