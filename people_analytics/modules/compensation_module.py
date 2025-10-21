# modules/compensation_module.py — v2.6
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.template_helper import render_download_template
from utils.uploader_helper import upload_data
from utils.pdf_helper import render_pdf_download_button

def run_compensation_module():
    st.markdown("""
    <div style="padding:18px;border-radius:10px;background:linear-gradient(90deg,#14532D,#22C55E);color:white;">
      <h2 style="margin:0">💰 Compensation Analytics</h2>
      <p style="margin:4px 0 0 0;">Pay structure, bonus, gender gap & market benchmarking.</p>
    </div>
    """, unsafe_allow_html=True)

    # template downloads (optional)
    emp_sample = pd.DataFrame([{"EmployeeID":"E1001","Gender":"Male","Department":"Finance","JobRole":"Analyst","JobLevel":"Analyst","CTC":600000,"Bonus":50000,"PerformanceRating":3}])
    bench_sample = pd.DataFrame([{"JobRole":"Analyst","JobLevel":"Analyst","MarketMedianCTC":650000}])
    c1,c2 = st.columns(2)
    with c1: render_download_template("Internal Template", emp_sample, "Internal_Template.csv")
    with c2: render_download_template("Benchmark Template", bench_sample, "Benchmark_Template.csv")

    emp_df = upload_data("Upload Internal Data (CSV/XLSX)")
    bench_df = upload_data("Upload Benchmark Data (optional) (CSV/XLSX)")

    if emp_df is None:
        st.info("Upload internal HR compensation data to continue.")
        return

    required = ["EmployeeID","Gender","Department","JobRole","JobLevel","CTC","Bonus","PerformanceRating"]
    missing = [c for c in required if c not in emp_df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    # compute bonus pct and basic metrics
    emp_df["CTC"] = pd.to_numeric(emp_df["CTC"], errors="coerce").fillna(0)
    emp_df["Bonus"] = pd.to_numeric(emp_df["Bonus"], errors="coerce").fillna(0)
    emp_df["BonusPct"] = (emp_df["Bonus"] / emp_df["CTC"].replace({0:pd.NA}))*100
    emp_df["BonusPct"] = emp_df["BonusPct"].fillna(0)

    st.dataframe(emp_df.head(), use_container_width=True)

    total = len(emp_df)
    avg_ctc = emp_df["CTC"].mean() if total else 0
    avg_bonuspct = emp_df["BonusPct"].mean() if total else 0
    c1,c2,c3 = st.columns(3)
    c1.metric("Avg CTC", f"₹{avg_ctc:,.0f}")
    c2.metric("Avg Bonus %", f"{avg_bonuspct:.1f}%")
    c3.metric("Employees", f"{total}")

    # Avg by level and bonus
    avg_ctc_by_level = emp_df.groupby("JobLevel", observed=True)["CTC"].mean().reset_index()
    avg_bonus_by_level = emp_df.groupby("JobLevel", observed=True)["BonusPct"].mean().reset_index()

    fig_ctc_level = px.bar(avg_ctc_by_level, x="JobLevel", y="CTC", text="CTC", title="Avg CTC by Job Level", color="JobLevel")
    fig_ctc_level.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
    fig_bonus_level = px.bar(avg_bonus_by_level, x="JobLevel", y="BonusPct", text="BonusPct", title="Avg Bonus % by Job Level", color="JobLevel")
    fig_bonus_level.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    st.plotly_chart(fig_ctc_level, use_container_width=True)
    st.plotly_chart(fig_bonus_level, use_container_width=True)

    # Gender gap
    gender_gap = emp_df.groupby("Gender", observed=True)["CTC"].mean().reset_index()
    fig_gender = px.bar(gender_gap, x="Gender", y="CTC", text="CTC", title="Avg CTC by Gender", color="Gender")
    fig_gender.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_gender, use_container_width=True)

    # Bonus vs CTC scatter
    fig_bonus_scatter = px.scatter(emp_df, x="CTC", y="BonusPct", hover_data=["EmployeeID","JobRole","JobLevel"], title="Bonus % vs CTC")
    st.plotly_chart(fig_bonus_scatter, use_container_width=True)

    # Market comparison if provided and valid
    comp_fig = None
    comp_df = None
    if bench_df is not None:
        if all(c in bench_df.columns for c in ["JobRole","JobLevel","MarketMedianCTC"]):
            bench_df["MarketMedianCTC"] = pd.to_numeric(bench_df["MarketMedianCTC"], errors="coerce").fillna(0)
            merged = emp_df.merge(bench_df, on=["JobRole","JobLevel"], how="left")
            merged["DiffPct"] = ((merged["CTC"] - merged["MarketMedianCTC"]) / merged["MarketMedianCTC"].replace({0:pd.NA}))*100
            merged["DiffPct"] = merged["DiffPct"].fillna(0)
            comp_df = merged.groupby("JobLevel", observed=True)[["CTC","MarketMedianCTC","DiffPct"]].mean().reset_index()
            comp_fig = px.bar(comp_df, x="JobLevel", y="DiffPct", text="DiffPct", title="Company vs Market Median (% Diff)", color="JobLevel")
            comp_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(comp_fig, use_container_width=True)
        else:
            st.warning("Benchmark file missing required columns or invalid. Skipping market comparison.")

    # PDF Export
    st.markdown("---")
    st.subheader("📄 Step 5 — Export Executive Report")
    data_blocks = [
        {"title":"Pay by Level","desc":"Avg CTC by job level","df":avg_ctc_by_level,"fig":fig_ctc_level,"insights":[f"Avg CTC: ₹{avg_ctc:,.0f}"]},
        {"title":"Bonus by Level","desc":"Avg bonus % by level","df":avg_bonus_by_level,"fig":fig_bonus_level,"insights":[f"Avg Bonus %: {avg_bonuspct:.1f}%"]},
        {"title":"Gender Pay Gap","desc":"Avg CTC by gender","df":gender_gap,"fig":fig_gender,"insights":[]},
        {"title":"Bonus vs CTC","desc":"Scatter of Bonus% vs CTC","df":emp_df[["EmployeeID","JobRole","JobLevel","CTC","BonusPct"]],"fig":fig_bonus_scatter,"insights":[]}
    ]
    if comp_df is not None and comp_fig is not None:
        data_blocks.append({"title":"Market Comparison","desc":"Internal vs Market medians","df":comp_df,"fig":comp_fig,"insights":[]})
    render_pdf_download_button("Compensation Analytics Executive Report","Compensation",data_blocks,"Compensation")