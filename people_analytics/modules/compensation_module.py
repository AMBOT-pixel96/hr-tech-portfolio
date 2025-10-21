# modules/compensation_module.py — v2.7 | Executive Polish Edition
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_compensation_module():
    st.markdown("""
    <div style="padding:20px;border-radius:12px;
                background:linear-gradient(90deg,#14532D,#22C55E);
                color:white;text-align:center;margin-bottom:20px;">
        <h2>💰 Compensation Analytics</h2>
        <p>Fairness, bonus trends, gender gap & market benchmarking.</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1: Uploads
    emp_df = upload_data("Upload Internal Data (CSV/XLSX)")
    bench_df = upload_data("Upload Benchmark Data (optional)")
    if emp_df is None:
        st.info("Please upload internal file to begin analysis.")
        return

    # Step 2: Validation
    required = ["EmployeeID","Gender","Department","JobRole","JobLevel","CTC","Bonus","PerformanceRating"]
    missing = [c for c in required if c not in emp_df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    # Step 3: Compute basics
    emp_df["CTC"] = pd.to_numeric(emp_df["CTC"], errors="coerce").fillna(0)
    emp_df["Bonus"] = pd.to_numeric(emp_df["Bonus"], errors="coerce").fillna(0)
    emp_df["BonusPct"] = (emp_df["Bonus"]/emp_df["CTC"].replace({0:pd.NA}))*100
    emp_df["BonusPct"] = emp_df["BonusPct"].fillna(0)

    total = len(emp_df)
    avg_ctc = emp_df["CTC"].mean()
    avg_bonus = emp_df["BonusPct"].mean()
    gender_gap = emp_df.groupby("Gender", observed=True)["CTC"].mean().reset_index()

    c1,c2,c3 = st.columns(3)
    c1.metric("Avg CTC", f"₹{avg_ctc:,.0f}")
    c2.metric("Avg Bonus %", f"{avg_bonus:.1f}%")
    c3.metric("Total Employees", total)

    # Step 4: Internal Pay visuals
    avg_ctc_level = emp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index()
    avg_bonus_level = emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index()

    fig_ctc = px.bar(avg_ctc_level, x="JobLevel", y="CTC", text="CTC", title="Avg CTC by Job Level", color="JobLevel")
    fig_ctc.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", marker_line_color='black', marker_line_width=1)

    fig_bonus = px.bar(avg_bonus_level, x="JobLevel", y="BonusPct", text="BonusPct", title="Avg Bonus % by Job Level", color="JobLevel")
    fig_bonus.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_color='black', marker_line_width=1)

    st.plotly_chart(fig_ctc, use_container_width=True)
    st.plotly_chart(fig_bonus, use_container_width=True)

    # Step 5: Gender Pay Gap
    fig_gender = px.bar(gender_gap, x="Gender", y="CTC", text="CTC", title="Avg CTC by Gender", color="Gender")
    fig_gender.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", marker_line_color='black', marker_line_width=1)
    st.plotly_chart(fig_gender, use_container_width=True)

    # Step 6: Benchmark Comparison (fixed)
    comp_summary = None
    fig_market = None
    if bench_df is not None:
        if "JobLevel" in bench_df.columns and "MarketMedianCTC" in bench_df.columns:
            bench_df["JobLevel"] = bench_df["JobLevel"].astype(str).str.strip()
            bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce").fillna(0)

            merged = emp_df.merge(bench_df[["JobLevel","MarketMedianCTC"]].drop_duplicates(), on="JobLevel", how="left")
            merged["DiffPct"] = ((merged["CTC"] - merged["MarketMedianCTC"]) / merged["MarketMedianCTC"].replace(0, pd.NA)) * 100
            merged["DiffPct"] = merged["DiffPct"].fillna(0).round(2)

            comp_summary = merged.groupby("JobLevel", observed=True)[["CTC","MarketMedianCTC","DiffPct"]].mean().reset_index()

            # Dual bar comparison
            comp_melt = comp_summary.melt(id_vars="JobLevel", value_vars=["CTC","MarketMedianCTC"], var_name="Type", value_name="Value")
            fig_market = px.bar(comp_melt, x="JobLevel", y="Value", color="Type", barmode="group",
                                text="Value", title="Internal vs Market Median by Job Level")
            fig_market.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", marker_line_color='black', marker_line_width=1)
            st.plotly_chart(fig_market, use_container_width=True)
        else:
            st.warning("Benchmark file missing 'JobLevel' or 'MarketMedianCTC'. Skipping comparison.")

    # Step 7: Export Report
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")

    data_blocks = [
        {"title":"CTC by Job Level","desc":"Average internal CTC by level","df":avg_ctc_level,"fig":fig_ctc,"insights":[f"Avg CTC ₹{avg_ctc:,.0f}"]},
        {"title":"Bonus by Job Level","desc":"Average bonus % by level","df":avg_bonus_level,"fig":fig_bonus,"insights":[f"Avg Bonus {avg_bonus:.1f}%"]},
        {"title":"Gender Pay Gap","desc":"Avg CTC by gender","df":gender_gap,"fig":fig_gender,"insights":[]}
    ]
    if comp_summary is not None and fig_market is not None:
        data_blocks.append({"title":"Internal vs Market","desc":"Company vs market median comparison","df":comp_summary,"fig":fig_market,"insights":[]})

    render_pdf_download_button("Compensation Analytics Executive Report","Compensation",data_blocks,"Compensation")